import {
  useState,
  useRef,
  useCallback,
  useEffect,
  type WheelEvent,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Move,
  Upload,
  Eye,
  EyeOff,
  Loader2,
  ImagePlus,
} from "lucide-react";
import type { Overlay } from "@/types/case";

interface XRayViewerProps {
  imageUrl?: string;
  overlays: Overlay[];
  onUploadImage?: (file: File) => Promise<void> | void;
  isUploading?: boolean;
}

export default function XRayViewer({
  imageUrl,
  overlays,
  onUploadImage,
  isUploading = false,
}: XRayViewerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [localImageUrl, setLocalImageUrl] = useState("");

  const panRef = useRef({
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });

  useEffect(() => {
    if (imageUrl) {
      setLocalImageUrl(imageUrl);
    }
  }, [imageUrl]);

  const displayedImage = localImageUrl || imageUrl || "";
  const hasImage = Boolean(displayedImage);
  const activeOverlays = overlays.filter((o) => o.visible);

  const reset = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const previewUrl = URL.createObjectURL(file);
    setLocalImageUrl(previewUrl);

    try {
      await onUploadImage?.(file);
    } catch (err) {
      console.error("Upload failed:", err);
    }

    e.target.value = "";
  };

  const handleWheel = useCallback(
    (e: WheelEvent<HTMLDivElement>) => {
      if (!hasImage) return;
      e.preventDefault();
      setScale((prev) => Math.min(5, Math.max(0.5, prev - e.deltaY * 0.001)));
    },
    [hasImage]
  );

  const handlePointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (!hasImage) return;

      panRef.current = {
        active: true,
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        originX: translate.x,
        originY: translate.y,
      };

      setIsPanning(true);
      e.currentTarget.setPointerCapture(e.pointerId);
    },
    [hasImage, translate.x, translate.y]
  );

  const handlePointerMove = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      const pan = panRef.current;
      if (!pan.active || pan.pointerId !== e.pointerId) return;

      const dx = e.clientX - pan.startX;
      const dy = e.clientY - pan.startY;

      setTranslate({
        x: pan.originX + dx,
        y: pan.originY + dy,
      });
    },
    []
  );

  const endPointerPan = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    if (pan.pointerId !== e.pointerId) return;

    panRef.current.active = false;
    setIsPanning(false);

    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  }, []);

  return (
    <div className="relative flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border bg-card/50 px-3 py-2">
        <button
          onClick={() => setScale((s) => Math.min(5, s + 0.25))}
          className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          title="Zoom In"
          type="button"
        >
          <ZoomIn size={18} />
        </button>

        <button
          onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}
          className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          title="Zoom Out"
          type="button"
        >
          <ZoomOut size={18} />
        </button>

        <button
          onClick={reset}
          className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          title="Reset View"
          type="button"
        >
          <RotateCcw size={18} />
        </button>

        <div className="mx-1 h-5 w-px bg-border" />

        <button
          onClick={handleUploadClick}
          className="flex items-center gap-2 rounded-md px-3 py-2 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          title="Upload X-Ray Image"
          type="button"
        >
          <Upload size={16} />
          <span className="hidden text-xs sm:inline">Upload</span>
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.webp,.dcm"
          className="hidden"
          onChange={handleFileChange}
        />

        <button
          onClick={() => setShowSegmentation((prev) => !prev)}
          className="flex items-center gap-1.5 rounded-md px-3 py-2 text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
          title="Show or hide segmentation"
          type="button"
        >
          {showSegmentation ? <EyeOff size={16} /> : <Eye size={16} />}
          <span className="hidden text-xs sm:inline">
            {showSegmentation ? "Hide Segmentation" : "Show Segmentation"}
          </span>
        </button>

        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          <Move size={14} />
          <span className="hidden sm:inline">{Math.round(scale * 100)}%</span>
        </div>
      </div>

      <div
        className={`relative flex-1 overflow-hidden bg-background select-none ${
          hasImage ? (isPanning ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
        }`}
        style={{ touchAction: "none" }}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endPointerPan}
        onPointerCancel={endPointerPan}
      >
        <div className="absolute inset-0 flex items-center justify-center">
          {hasImage ? (
            <div
              className="relative"
              style={{
                transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
                transformOrigin: "center center",
                transition: isPanning ? "none" : "transform 0.05s linear",
                willChange: "transform",
              }}
            >
              <img
                src={displayedImage}
                alt="Chest X-ray"
                className="pointer-events-none block max-h-[85vh] max-w-[85vw] object-contain"
                draggable={false}
              />

              {showSegmentation &&
                activeOverlays.map((overlay) => (
                  <img
                    key={overlay.id}
                    src={overlay.dataUrl}
                    alt={overlay.name}
                    className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                    style={{ opacity: overlay.opacity }}
                    draggable={false}
                  />
                ))}

              {isUploading && (
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-background/40 backdrop-blur-sm">
                  <div className="flex flex-col items-center rounded-2xl border border-border bg-card/95 px-6 py-5 shadow-xl">
                    <div className="relative h-20 w-20">
                      <svg className="h-20 w-20 -rotate-90" viewBox="0 0 100 100">
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          className="text-muted/30"
                        />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          strokeLinecap="round"
                          strokeDasharray="188.5"
                          strokeDashoffset="56"
                          className="text-primary animate-pulse"
                        />
                      </svg>

                      <div className="absolute inset-0 flex items-center justify-center">
                        <Loader2 size={22} className="animate-spin text-primary" />
                      </div>
                    </div>

                    <p className="mt-4 text-sm font-semibold text-foreground">
                      Segmenting image...
                    </p>
                    <p className="mt-1 text-center text-xs text-muted-foreground">
                      Please wait while the backend processes the X-ray
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="pointer-events-none flex h-[400px] w-[600px] flex-col items-center justify-center rounded-2xl border border-dashed border-border bg-card/30 text-center">
              <div className="mb-4 rounded-full border border-border bg-card p-4">
                <ImagePlus size={28} className="text-muted-foreground" />
              </div>
              <p className="text-lg font-medium text-foreground">
                Upload an image to start
              </p>
              <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                Select a chest X-ray image to begin viewing, segmentation, and technical evaluation.
              </p>
              <button
                onClick={handleUploadClick}
                type="button"
                className="pointer-events-auto mt-5 inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
              >
                <Upload size={16} />
                Choose Image
              </button>
            </div>
          )}
        </div>
      </div>

      {hasImage && showSegmentation && activeOverlays.length > 0 && (
        <div className="flex items-center gap-3 border-t border-border bg-card/50 px-3 py-2">
          <span className="text-xs text-muted-foreground">Active:</span>
          {activeOverlays.map((o) => (
            <div key={o.id} className="flex items-center gap-1.5">
              <div
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: o.color }}
              />
              <span className="text-xs text-secondary-foreground">{o.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}