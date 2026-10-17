import type { CXRCase } from "@/types/case";

export const mockCases: CXRCase[] = [
  {
    caseId: "cxr_001",
    caseTitle: "PA Chest Radiograph — Mild Rotation",
    description: "Adult PA chest radiograph demonstrating mild patient rotation with otherwise acceptable positioning.",
    imageUrl: "/images/sample-cxr.jpg",
    overlays: [
      {
        id: "rotation",
        name: "Rotation",
        visible: false,
        type: "contour",
        opacity: 0.7,
        color: "#a855f7",
        dataUrl: "/overlays/rotation.png",
      },
      {
        id: "clavicles",
        name: "Clavicles",
        visible: false,
        type: "mask",
        opacity: 0.6,
        color: "#ec4899",
        dataUrl: "/overlays/clavicles.png",
      },
      {
        id: "spinous_process",
        name: "Spinous Process",
        visible: false,
        type: "contour",
        opacity: 0.7,
        color: "#06b6d4",
        dataUrl: "/overlays/spinous.png",
      },
      {
        id: "tilting",
        name: "Tilting",
        visible: false,
        type: "annotation",
        opacity: 0.65,
        color: "#f59e0b",
        dataUrl: "/overlays/tilting.png",
      },
    ],
    report: {
      mode: "technical",
      rawText: `PACEMAN TECHNICAL EVALUATION — PA CHEST RADIOGRAPH

Case: cxr_001
Modality: Computed Radiography
Projection: PA Erect

POSITION
The patient demonstrates mild rotation toward the left side. The medial ends of the clavicles are not equidistant from the spinous processes of the thoracic vertebrae, with approximately 5mm discrepancy. The shoulders are adequately depressed and internally rotated, with the scapulae projected clear of the lung fields bilaterally. No significant tilt is identified. The patient is not in a lordotic or kyphotic position. Adequate inspiration is demonstrated with 10 posterior ribs visible above the diaphragm on the right side.

AREA COVERAGE
The chest radiograph demonstrates adequate coverage. The apices, costophrenic angles, and lateral chest walls are included. The T1 vertebra is visualised. Both costophrenic angles are sharp and clearly demonstrated.

CENTERING
The central ray appears correctly centred at the level of T6, which is appropriate for a PA chest projection. No obvious decentering artefact is present.

COLLIMATION
Appropriate collimation has been applied. The collimation borders are visible at the edges of the image, indicating that the field size has been restricted to the area of clinical interest.

EXPOSURE
Exposure factors appear appropriate. The thoracic vertebral bodies are faintly visible through the cardiac shadow, which is the expected standard for a PA chest radiograph. There is no overexposure or underexposure.

MARKER
A left anatomical side marker is present and correctly placed outside the area of clinical interest.

ARTEFACTS
No significant artefacts are identified on this image.

FINAL CONCLUSION
This is a clinically diagnostic and acceptable PA chest radiograph with mild rotation as the only notable positioning error. The image quality is sufficient for diagnostic interpretation.`,
      technicalSections: [
        {
          title: "POSITION",
          subsections: [
            {
              title: "Rotation",
              content: [
                "The patient demonstrates mild rotation toward the left side.",
                "The medial ends of the clavicles are not equidistant from the spinous processes of the thoracic vertebrae, with approximately 5mm discrepancy.",
                "This degree of rotation is unlikely to significantly affect diagnostic interpretation.",
              ],
              severity: "mild",
            },
            {
              title: "Shoulder Position / Tension",
              content: [
                "The shoulders are adequately depressed and internally rotated.",
                "The scapulae are projected clear of the lung fields bilaterally.",
                "This allows for unobstructed visualisation of the lateral lung zones.",
              ],
            },
            {
              title: "Tilt",
              content: [
                "No significant tilt is identified.",
                "The clavicles appear symmetrical in their vertical positioning relative to the lung apices.",
              ],
            },
            {
              title: "Lordotic / Kyphotic Positioning",
              content: [
                "The patient is not in a lordotic or kyphotic position.",
                "The clavicles project over the lung apices as expected for a true PA projection.",
              ],
            },
            {
              title: "Inspiration",
              content: [
                "Adequate inspiration is demonstrated.",
                "10 posterior ribs are visible above the right hemidiaphragm.",
                "This meets the standard criterion of 9–10 posterior ribs for adequate inspiration.",
              ],
            },
            {
              title: "Scapulae Positioning",
              content: [
                "Both scapulae are adequately rotated clear of the lung fields.",
                "Internal rotation of the arms has been successfully achieved.",
              ],
            },
          ],
        },
        {
          title: "AREA COVERAGE",
          subsections: [
            {
              title: "Coverage Assessment",
              content: [
                "The chest radiograph demonstrates adequate coverage.",
                "Lung apices, costophrenic angles, and lateral chest walls are all included.",
                "The T1 vertebra is visualised superiorly.",
                "Both costophrenic angles are sharp and clearly demonstrated.",
              ],
            },
          ],
        },
        {
          title: "CENTERING",
          subsections: [
            {
              title: "Central Ray Position",
              content: [
                "The central ray appears correctly centred at the level of T6.",
                "This is the appropriate centering point for a PA chest projection.",
                "No obvious decentering artefact is present.",
              ],
            },
          ],
        },
        {
          title: "COLLIMATION",
          subsections: [
            {
              title: "Field Restriction",
              content: [
                "Appropriate collimation has been applied.",
                "Collimation borders are visible at the image edges.",
                "The field size has been restricted to the area of clinical interest, reducing patient dose.",
              ],
            },
          ],
        },
        {
          title: "EXPOSURE",
          subsections: [
            {
              title: "Exposure Assessment",
              content: [
                "Exposure factors appear appropriate.",
                "The thoracic vertebral bodies are faintly visible through the cardiac shadow.",
                "This meets the expected standard for a PA chest radiograph.",
                "No overexposure or underexposure is identified.",
              ],
            },
          ],
        },
        {
          title: "MARKER",
          subsections: [
            {
              title: "Marker Placement",
              content: [
                "A left anatomical side marker is present.",
                "The marker is correctly placed outside the area of clinical interest.",
              ],
            },
          ],
        },
        {
          title: "ARTEFACTS",
          subsections: [
            {
              title: "Artefact Assessment",
              content: [
                "No significant artefacts are identified on this image.",
                "The image is free from motion blur, foreign bodies, and processing artefacts.",
              ],
            },
          ],
        },
        {
          title: "PRACTICAL CORRECTIONS",
          subsections: [
            {
              title: "Recommendations",
              content: [
                "To correct the mild rotation, ensure the patient's shoulders are equidistant from the image receptor.",
                "Palpate the posterior aspect of the shoulders to confirm equal contact with the bucky.",
                "Use a mirror or verbal coaching to help the patient maintain a true PA position.",
              ],
              severity: "mild",
            },
          ],
        },
        {
          title: "STRENGTHS & WEAKNESSES",
          subsections: [
            {
              title: "Strengths",
              content: [
                "Excellent inspiration with 10 posterior ribs demonstrated.",
                "Scapulae well-cleared from lung fields.",
                "Appropriate exposure and collimation.",
                "Clear, sharp costophrenic angles.",
              ],
            },
            {
              title: "Weaknesses",
              content: [
                "Mild rotation toward the patient's left.",
                "This is the only notable positioning error.",
              ],
              severity: "mild",
            },
          ],
        },
        {
          title: "FINAL CONCLUSION",
          subsections: [
            {
              title: "Overall Assessment",
              content: [
                "This is a clinically diagnostic and acceptable PA chest radiograph.",
                "The mild rotation is the only notable positioning error and is unlikely to significantly impact diagnostic interpretation.",
                "The overall image quality is good and suitable for clinical use.",
              ],
            },
          ],
        },
      ],
    },
  },
];
