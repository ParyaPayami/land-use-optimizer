#!/usr/bin/env python3
"""
Generate a highly professional, well-formatted Word (.docx) document 
containing the Title Page, Cover Letter, and Generative AI Disclosure Statement
for the CEUS first submission.
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = Document()
    
    # Define professional styles & margins (1 inch on all sides)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Helper function to add headings with professional styling
    def add_custom_heading(text, level, space_before=12, space_after=6):
        heading = doc.add_heading(text, level=level)
        heading.paragraph_format.space_before = Pt(space_before)
        heading.paragraph_format.space_after = Pt(space_after)
        heading.paragraph_format.keep_with_next = True
        run = heading.runs[0]
        run.font.name = 'Calibri'
        run.font.color.rgb = RGBColor(31, 78, 121)  # Deep professional blue
        return heading

    # Helper for standard body text
    def add_custom_paragraph(text, bold_prefix=None, space_after=6, line_spacing=1.15):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = line_spacing
        
        if bold_prefix:
            run_bold = p.add_run(bold_prefix)
            run_bold.font.name = 'Calibri'
            run_bold.font.size = Pt(11)
            run_bold.bold = True
            
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        return p

    # =========================================================================
    # DOCUMENT 1: TITLE PAGE (AUTHOR DETAILS)
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(12)
    run_title = title_p.add_run("PIMALUOS: An Open-Source Physics-Informed Multi-Agent Framework for Urban Land-Use Optimization")
    run_title.font.name = 'Calibri'
    run_title.font.size = Pt(18)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(24)
    run_sub = sub_p.add_run("Title Page with Complete Author Details (Separate File for Peer Review)")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(12)
    run_sub.italic = True

    # Author details
    add_custom_paragraph("Parya Payami", bold_prefix="Author: ").alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_custom_paragraph("School of Architecture and Urban Planning, University of Wisconsin Milwaukee, Milwaukee, WI, USA", bold_prefix="Affiliation: ").alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_custom_paragraph("ppayami@uwm.edu", bold_prefix="Corresponding Email: ").alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_custom_heading("Acknowledgements", level=2, space_before=24)
    add_custom_paragraph(
        "The author thanks the School of Architecture and Urban Planning at the University of Wisconsin "
        "Milwaukee for academic support. The author also thanks the open-source urban data science "
        "community for the foundational tools, including OSMnx, PyTorch Geometric, and LangChain, that "
        "made this integrated framework possible."
    )

    doc.add_page_break()

    # =========================================================================
    # DOCUMENT 2: COVER LETTER FOR CEUS SUBMISSION
    # =========================================================================
    date_p = doc.add_paragraph()
    date_p.paragraph_format.space_after = Pt(12)
    run_date = date_p.add_run("May 31, 2026")
    run_date.font.name = 'Calibri'
    run_date.font.size = Pt(11)

    recipient_p = doc.add_paragraph()
    recipient_p.paragraph_format.space_after = Pt(18)
    recipient_p.paragraph_format.line_spacing = 1.15
    r_run = recipient_p.add_run(
        "To: The Editor-in-Chief and Guest Editors\n"
        "Special Issue on Open Urban Data Science\n"
        "Computers, Environment and Urban Systems (CEUS)"
    )
    r_run.font.name = 'Calibri'
    r_run.font.size = Pt(11)
    r_run.bold = True

    subj_p = doc.add_paragraph()
    subj_p.paragraph_format.space_after = Pt(12)
    s_run = subj_p.add_run("Subject: Submission of Original Manuscript — PIMALUOS")
    s_run.font.name = 'Calibri'
    s_run.font.size = Pt(11)
    s_run.bold = True

    add_custom_paragraph("Dear Editors,")
    add_custom_paragraph(
        "I am pleased to submit my original manuscript titled \"PIMALUOS: An Open-Source Physics-Informed "
        "Multi-Agent Framework for Urban Land-Use Optimization\" for consideration in the Special Issue on "
        "Open Urban Data Science in Computers, Environment and Urban Systems (CEUS)."
    )
    add_custom_paragraph(
        "This work represents an effort in quantifying the impact of urban planning and design decision-making "
        "by combining heterogeneous parcel graphs, zoning regulations, stakeholder agents, and simplified "
        "physical-capacity checks. In addition to serving as a modular research twin, the software acts as an "
        "assist to developers and planning agencies to find optimized land-use layouts that satisfy municipal "
        "infrastructure capacity thresholds."
    )
    add_custom_paragraph(
        "I confirm that this manuscript is my original contribution, is not currently under consideration for "
        "publication elsewhere, and has been fully approved for submission. All related code, reproducible "
        "environment setups, and datasets are open-source and publicly accessible under the MIT license."
    )
    add_custom_paragraph("Thank you for your time and consideration of my work.")

    sign_p = doc.add_paragraph()
    sign_p.paragraph_format.space_before = Pt(12)
    sign_p.paragraph_format.space_after = Pt(24)
    run_sign = sign_p.add_run(
        "Sincerely,\n\n"
        "Parya Payami\n"
        "School of Architecture and Urban Planning\n"
        "University of Wisconsin Milwaukee\n"
        "ppayami@uwm.edu"
    )
    run_sign.font.name = 'Calibri'
    run_sign.font.size = Pt(11)

    doc.add_page_break()

    # =========================================================================
    # DOCUMENT 3: ELSEVIER GENERATIVE AI DECLARATION
    # =========================================================================
    add_custom_heading("Declaration of Generative AI Use in Manuscript Preparation", level=1, space_before=12, space_after=12)
    
    add_custom_paragraph(
        "In accordance with Elsevier's Generative AI and AI-assisted Technologies in the Manuscript Preparation Process policy, "
        "the following declaration outlines the support utilized during the preparation of this work."
    )

    add_custom_heading("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", level=2, space_before=18)
    
    add_custom_paragraph(
        "During the preparation of this work the author(s) used Antigravity Gemini, ChatGPT, Claude, and Grok "
        "in order to support coding and formatting improvements. After using this tool/service, the "
        "author(s) reviewed and edited the content as needed and take(s) full responsibility for the content "
        "of the published article.",
        bold_prefix="Statement: "
    )

    doc.add_page_break()

    # =========================================================================
    # DOCUMENT 4: RESEARCH HIGHLIGHTS
    # =========================================================================
    add_custom_heading("Research Highlights", level=1, space_before=12, space_after=12)
    
    add_custom_paragraph(
        "Highlights consist of a short collection of bullet points that convey the core findings and provide "
        "readers with a quick textual overview of the article. They are brief, single-sentence points that "
        "summarize the paper. In accordance with Computers, Environment and Urban Systems (CEUS) / Elsevier "
        "guidelines, these are 5 highlights, each strictly within the 85-character limit (including spaces):"
    )

    highlights = [
        "An open-source Python framework integrates GNNs, RAG-LLMs, and multi-agent systems.",
        "Models urban parcels as a heterogeneous graph with five distinct edge types.",
        "A RAG-LLM parser digitizes complex zoning codes with over 96% extraction accuracy.",
        "MARL agents negotiate multi-objective land-use and stakeholder trade-offs.",
        "Validated on Manhattan's 42,075 parcels, satisfying infrastructure limits."
    ]

    for highlight in highlights:
        p_bullet = doc.add_paragraph(style='List Bullet')
        p_bullet.paragraph_format.space_after = Pt(8)
        p_bullet.paragraph_format.line_spacing = 1.15
        run_bullet = p_bullet.add_run(highlight)
        run_bullet.font.name = 'Calibri'
        run_bullet.font.size = Pt(11)

    output_path = Path("./Submission_Documents.docx")
    doc.save(output_path)
    print(f"Successfully generated complete Word submission document at {output_path.resolve()}!")

if __name__ == "__main__":
    main()
