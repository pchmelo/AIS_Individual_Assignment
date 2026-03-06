import json
import os
import re
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT


def _create_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c5282'),
    ))
    
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568'),
    ))
    
    styles.add(ParagraphStyle(
        name='StageHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5282'),
        borderPadding=5,
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading3'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#3182ce'),
    ))
    
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading4'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#4a6fa5'),
    ))
    
    styles['BodyText'].fontSize = 10
    styles['BodyText'].spaceAfter = 8
    styles['BodyText'].alignment = TA_JUSTIFY
    styles['BodyText'].leading = 14
    
    styles.add(ParagraphStyle(
        name='BulletPoint',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=20,
        spaceAfter=6,
        bulletIndent=10,
        leading=13,
    ))
    
    styles.add(ParagraphStyle(
        name='NestedBullet',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=40,
        spaceAfter=4,
        bulletIndent=30,
        leading=12,
    ))
    
    styles.add(ParagraphStyle(
        name='KeyValue',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=15,
        spaceAfter=4,
        leading=13,
    ))
    
    styles['Code'].fontSize = 8
    styles['Code'].fontName = 'Courier'
    styles['Code'].leftIndent = 10
    styles['Code'].rightIndent = 10
    styles['Code'].spaceAfter = 8
    styles['Code'].backColor = colors.HexColor('#f5f5f5')
    styles['Code'].borderPadding = 5
    
    return styles


def _parse_markdown_sections(report_text: str) -> dict:
    """Parse markdown report into sections."""
    sections = {}
    
    lines = report_text.split('\n')
    
    metadata = {}
    in_metadata = False
    for line in lines[:20]:
        if line.strip() == '## Metadata':
            in_metadata = True
            continue
        if in_metadata and line.startswith('- **'):
            match = re.match(r'-\s*\*\*(.+?):\*\*\s*(.+)', line)
            if match:
                key = match.group(1).lower().replace(' ', '_')
                value = match.group(2).strip()
                metadata[key] = value
        if line.strip() == '---':
            in_metadata = False
    
    if metadata:
        sections['metadata'] = metadata
    
    current_section = None
    current_content = []
    
    for line in lines:
        if line.startswith('## ') and not line.startswith('## Metadata'):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    return sections


def _clean_text_for_pdf(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?tool_call>', '', text)
    text = re.sub(r'<function=[^>]*>.*?</function>', '', text, flags=re.DOTALL)
    text = re.sub(r'<parameter=[^>]*>.*?</parameter>', '', text, flags=re.DOTALL)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _escape_xml(text: str) -> str:
    """Escape XML special chars for reportlab."""
    if not text:
        return ""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def _format_column_table(lines: list, styles) -> list:
    """Format 'Column: X | Reason: Y | Values: Z' lines as a proper table."""
    flowables = []
    table_data = [['Column', 'Description', 'Sample Values']]
    
    for line in lines:
        # Match pattern: Column: X | Reason: Y | Values: [Z] or Values: Z
        match = re.match(r'Column:\s*(.+?)\s*\|\s*Reason:\s*(.+?)\s*\|\s*Values:\s*\[?([^\]]+)\]?', line)
        if match:
            col_name = match.group(1).strip()
            reason = match.group(2).strip()
            values = match.group(3).strip()
            # Truncate long values
            if len(values) > 35:
                values = values[:32] + "..."
            # Truncate long reasons
            if len(reason) > 50:
                reason = reason[:47] + "..."
            table_data.append([col_name, reason, values])
    
    if len(table_data) > 1:
        # Use Paragraph objects for proper text wrapping
        para_table_data = []
        for i, row in enumerate(table_data):
            para_row = []
            for j, cell in enumerate(row):
                cell_style = ParagraphStyle(
                    'TableCell',
                    parent=styles['BodyText'],
                    fontSize=8,
                    leading=10,
                    alignment=TA_LEFT,
                )
                if i == 0:  # Header row
                    cell_style.fontName = 'Helvetica-Bold'
                    cell_style.textColor = colors.white
                para_row.append(Paragraph(_escape_xml(str(cell)), cell_style))
            para_table_data.append(para_row)
        
        table = Table(para_table_data, colWidths=[1.4*inch, 2.6*inch, 2.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a6fa5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 0.15*inch))
    
    return flowables


def _format_markdown_table(table_lines: list, styles) -> list:
    """Parse markdown table lines and convert to PDF table."""
    flowables = []
    if not table_lines:
        return flowables
    
    table_data = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.split('|')]
        cells = [c for c in cells if c]
        
        if not cells:
            continue
        
        # Skip separator lines - all cells are just dashes/colons
        is_separator = all(re.match(r'^[\-:]+$', cell.strip()) for cell in cells)
        if is_separator:
            continue
        
        cleaned_cells = []
        for cell in cells:
            cell = re.sub(r'\*\*(.+?)\*\*', r'\1', cell)
            cell = re.sub(r'<br\s*/?>', '\n', cell)
            cell = cell.strip()
            cleaned_cells.append(cell)
        table_data.append(cleaned_cells)
    
    if len(table_data) >= 1:
        num_cols = max(len(row) for row in table_data) if table_data else 1
        available_width = 6.5 * inch
        col_width = available_width / num_cols
        col_widths = [min(col_width, 2.5*inch) for _ in range(num_cols)]
        
        for row in table_data:
            while len(row) < num_cols:
                row.append('')
        
        para_table_data = []
        for i, row in enumerate(table_data):
            para_row = []
            for cell in row:
                style = styles['BodyText'] if i > 0 else styles['BodyText']
                style_copy = ParagraphStyle(
                    'TableCell',
                    parent=style,
                    fontSize=8,
                    leading=10,
                    alignment=TA_LEFT,
                )
                escaped_cell = _escape_xml(cell)
                escaped_cell = escaped_cell.replace('\n', '<br/>')
                para_row.append(Paragraph(escaped_cell, style_copy))
            para_table_data.append(para_row)
        
        table = Table(para_table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a6fa5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ]))
        flowables.append(table)
        flowables.append(Spacer(1, 0.15*inch))
    
    return flowables


def _format_markdown_to_paragraphs(text: str, styles) -> list:
    """Convert markdown-style text to reportlab flowables."""
    flowables = []
    lines = text.split('\n')
    
    current_paragraph = []
    in_code_block = False
    code_block_content = []
    column_lines = []
    markdown_table_lines = []
    
    def flush_paragraph():
        nonlocal current_paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            para_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', para_text)
            para_text = _escape_xml(para_text)
            # Re-apply bold tags after escaping
            para_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', para_text)
            flowables.append(Paragraph(para_text, styles['BodyText']))
            current_paragraph = []
    
    def flush_column_lines():
        nonlocal column_lines
        if column_lines:
            flowables.extend(_format_column_table(column_lines, styles))
            column_lines = []
    
    def flush_markdown_table():
        nonlocal markdown_table_lines
        if markdown_table_lines:
            flowables.extend(_format_markdown_table(markdown_table_lines, styles))
            markdown_table_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if stripped.startswith('```'):
            flush_paragraph()
            flush_column_lines()
            flush_markdown_table()
            if in_code_block:
                if code_block_content:
                    code_text = '\n'.join(code_block_content)
                    escaped = _escape_xml(code_text)
                    flowables.append(Paragraph(escaped.replace('\n', '<br/>'), styles['Code']))
                    code_block_content = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue
        
        if in_code_block:
            code_block_content.append(line)
            i += 1
            continue
        
        if stripped.startswith('|') and stripped.endswith('|'):
            flush_paragraph()
            flush_column_lines()
            markdown_table_lines.append(stripped)
            i += 1
            continue
        elif markdown_table_lines:
            flush_markdown_table()
        
        # Match Column lines with optional numbered prefix and bold markers 
        # (e.g., "1.  **Column:** X | Reason: Y" or "1.  Column: X | Reason: Y")
        column_match = re.match(r'^(?:\d+\.\s*)?(?:\*\*)?Column:(?:\*\*)?\s*.+?\s*\|\s*Reason:', stripped)
        if column_match:
            flush_paragraph()
            # Extract the Column part, removing number prefix and bold markers
            clean_line = re.sub(r'^\d+\.\s*', '', stripped)
            clean_line = clean_line.replace('**Column:**', 'Column:').replace('**', '')
            column_lines.append(clean_line)
            i += 1
            continue
        elif column_lines:
            flush_column_lines()
        
        if stripped.startswith('#### '):
            flush_paragraph()
            header_text = stripped[5:]
            # Convert markdown bold to HTML bold tags
            header_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', header_text)
            header_text = _escape_xml(header_text)
            header_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', header_text)
            flowables.append(Paragraph(header_text, styles['SubsectionHeader']))
            i += 1
            continue
        
        if stripped.startswith('### '):
            flush_paragraph()
            header_text = stripped[4:]
            # Convert markdown bold to HTML bold tags
            header_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', header_text)
            header_text = _escape_xml(header_text)
            header_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', header_text)
            flowables.append(Paragraph(header_text, styles['SectionHeader']))
            i += 1
            continue
        
        if stripped.startswith('## '):
            flush_paragraph()
            header_text = stripped[3:]
            # Convert markdown bold to HTML bold tags
            header_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', header_text)
            header_text = _escape_xml(header_text)
            header_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', header_text)
            flowables.append(Paragraph(header_text, styles['SectionHeader']))
            i += 1
            continue
        
        # Handle standalone bold lines (e.g., **Mitigation Strategies:** or **Title:**)
        # Just strip the ** markers and render as italic subsection header
        bold_only_match = re.match(r'^\*\*(.+?)\*\*:?$', stripped)
        if bold_only_match:
            flush_paragraph()
            header_text = bold_only_match.group(1).strip()
            flowables.append(Spacer(1, 0.1*inch))
            flowables.append(Paragraph(f'<i>{_escape_xml(header_text)}:</i>', styles['BodyText']))
            i += 1
            continue
        
        # Skip [TOOL RESULT] sections
        if re.match(r'^\[TOOL RESULT\]', stripped, re.IGNORECASE):
            flush_paragraph()
            i += 1
            brace_count = 0
            while i < len(lines):
                skip_line = lines[i].strip()
                brace_count += skip_line.count('{') - skip_line.count('}')
                if brace_count <= 0 and (skip_line.startswith('[') or skip_line.startswith('#')):
                    break
                if brace_count <= 0 and not skip_line:
                    i += 1
                    break
                i += 1
            continue
        
        if re.match(r'^\[.+\]$', stripped) and '-' not in stripped:
            flush_paragraph()
            header_text = stripped[1:-1].replace('_', ' ').title()
            flowables.append(Spacer(1, 0.1*inch))
            flowables.append(Paragraph(_escape_xml(header_text), styles['SectionHeader']))
            i += 1
            continue
        
        if re.match(r'^-{3,}$', stripped):
            i += 1
            continue
        
        if re.match(r'^[\t ]+[-*] ', line):
            flush_paragraph()
            bullet_content = re.sub(r'^[\t ]+[-*] ', '', line)
            bullet_text = '  ◦ ' + bullet_content.strip()
            bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet_text)
            bullet_text = _escape_xml(bullet_text)
            bullet_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', bullet_text)
            flowables.append(Paragraph(bullet_text, styles['NestedBullet']))
            i += 1
            continue
        
        if stripped.startswith('- ') or stripped.startswith('* '):
            flush_paragraph()
            bullet_text = '• ' + stripped[2:]
            bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet_text)
            bullet_text = _escape_xml(bullet_text)
            bullet_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', bullet_text)
            flowables.append(Paragraph(bullet_text, styles['BulletPoint']))
            i += 1
            continue
        
        if re.match(r'^\d+\.\s', stripped):
            flush_paragraph()
            list_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', stripped)
            list_text = _escape_xml(list_text)
            list_text = re.sub(r'&lt;b&gt;(.+?)&lt;/b&gt;', r'<b>\1</b>', list_text)
            flowables.append(Paragraph(list_text, styles['BulletPoint']))
            i += 1
            continue
        
        if not stripped:
            flush_paragraph()
            i += 1
            continue
        
        current_paragraph.append(stripped)
        i += 1
    
    flush_paragraph()
    flush_column_lines()
    flush_markdown_table()
    
    return flowables


def generate_pdf_bytes(report_path: str) -> bytes:
    """Generate PDF from markdown report file."""
    if not os.path.exists(report_path):
        md_path = report_path.replace('.txt', '.md')
        if os.path.exists(md_path):
            report_path = md_path
        else:
            raise FileNotFoundError(f"Report not found: {report_path}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    sections = _parse_markdown_sections(report_text)
    styles = _create_styles()
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )
    
    flowables = []
    flowables.append(Paragraph("Dataset Fairness Evaluation Report", styles['ReportTitle']))
    
    if 'metadata' in sections:
        meta = sections['metadata']
        flowables.append(Paragraph(f"Dataset: {meta.get('dataset', 'N/A')}", styles['Subtitle']))
        flowables.append(Paragraph(f"Target Column: {meta.get('target_column', 'N/A')}", styles['Subtitle']))
        flowables.append(Paragraph(f"Generated: {meta.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}", styles['Subtitle']))
    
    flowables.append(Spacer(1, 0.5*inch))
    flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    flowables.append(Spacer(1, 0.3*inch))
    
    for section_name, section_content in sections.items():
        if section_name == 'metadata' or not section_content:
            continue
        
        flowables.append(Paragraph(section_name, styles['StageHeader']))
        flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e0e0e0')))
        flowables.append(Spacer(1, 0.1*inch))
        
        clean_content = _clean_text_for_pdf(section_content)
        para_flowables = _format_markdown_to_paragraphs(clean_content, styles)
        flowables.extend(para_flowables)
        
        flowables.append(Spacer(1, 0.2*inch))
    
    flowables.append(Spacer(1, 0.5*inch))
    flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    flowables.append(Spacer(1, 0.2*inch))
    flowables.append(Paragraph(
        "Report generated by Dataset Fairness Evaluation System",
        styles['Subtitle']
    ))
    
    doc.build(flowables)
    
    buffer.seek(0)
    return buffer.getvalue()
