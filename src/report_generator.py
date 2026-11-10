from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.units import mm
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


def generate_pdf_report(
    patient,
    prediction,
    explanation,
    report_id=None
):
    """
    Generate a professional KidneyCare AI PDF report.

    Parameters:
        patient: dictionary containing patient information
        prediction: dictionary containing prediction information
        explanation: pandas DataFrame containing SHAP explanations

    Returns:
        Path to generated PDF file
    """

    if report_id is None:
        report_id = datetime.now().strftime("%Y%m%d%H%M%S")

    filename = f"KidneyCare_AI_Report_{report_id}.pdf"
    output_path = REPORTS_DIR / filename

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=24,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=18,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=10,
        spaceAfter=8,
    )

    normal_style = ParagraphStyle(
        "NormalReport",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=14,
    )

    story = []

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "KidneyCare AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Explainable Artificial Intelligence-Based "
            "Clinical Decision Support System",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}",
            subtitle_style
        )
    )

    # ---------------------------------------------------------
    # PATIENT INFORMATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "1. Patient Information",
            heading_style
        )
    )

    patient_data = [
        ["Patient ID", str(patient.get("patient_id", "N/A"))],
        ["Full Name", str(patient.get("full_name", "N/A"))],
        ["Age", str(patient.get("age", "N/A"))],
        ["Gender", str(patient.get("gender", "N/A"))],
        ["Phone", str(patient.get("phone", "N/A"))],
    ]

    patient_table = Table(
        patient_data,
        colWidths=[45 * mm, 125 * mm]
    )

    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(patient_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "2. CKD Prediction Result",
            heading_style
        )
    )

    probability = prediction.get("probability", 0)

    try:
        probability_percentage = float(probability) * 100
    except (TypeError, ValueError):
        probability_percentage = 0

    prediction_value = prediction.get(
        "prediction",
        "N/A"
    )

    model_name = prediction.get(
        "model",
        "Random Forest"
    )

    prediction_data = [
        ["Prediction", str(prediction_value)],
        [
            "CKD Probability",
            f"{probability_percentage:.2f}%"
        ],
        ["Model Used", str(model_name)],
        [
            "Prediction Date",
            datetime.now().strftime(
                "%d %B %Y, %H:%M"
            )
        ],
    ]

    prediction_table = Table(
        prediction_data,
        colWidths=[55 * mm, 115 * mm]
    )

    prediction_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(prediction_table)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # INTERPRETATION
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "3. AI Explanation",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "The following features had the largest "
            "SHAP contributions to the model prediction. "
            "Positive contributions indicate that the feature "
            "increased the model's CKD prediction, while negative "
            "contributions indicate that the feature decreased "
            "the model's CKD prediction.",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    explanation_data = [
        [
            "Feature",
            "Value",
            "SHAP Value",
            "Contribution"
        ]
    ]

    # Only include the top 10 explanations
    top_explanations = explanation.head(10)

    for _, row in top_explanations.iterrows():

        shap_value = float(row["shap_value"])

        explanation_data.append([
            str(row["feature"]),
            str(row["feature_value"]),
            f"{shap_value:.4f}",
            str(row["contribution"]),
        ])

    explanation_table = Table(
        explanation_data,
        colWidths=[
            45 * mm,
            30 * mm,
            30 * mm,
            65 * mm
        ],
        repeatRows=1
    )

    explanation_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.append(explanation_table)
    story.append(Spacer(1, 15))

    # ---------------------------------------------------------
    # DISCLAIMER
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            "4. Important Notice",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "<b>Academic Prototype:</b> KidneyCare AI is an "
            "academic clinical decision-support prototype. "
            "The prediction generated by this system should "
            "not be treated as a medical diagnosis. Qualified "
            "healthcare professionals should interpret the "
            "results together with other clinical information "
            "and appropriate medical assessment.",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Generated by KidneyCare AI",
            subtitle_style
        )
    )

    doc.build(story)

    return output_path