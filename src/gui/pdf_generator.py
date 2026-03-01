import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Optional

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
        textColor=colors.HexColor('#1a1a2e'),
    ))
    
    styles.add(ParagraphStyle(
        name='Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a4a6a'),
    ))
    
    styles.add(ParagraphStyle(
        name='StageHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#16213e'),
        borderPadding=5,
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading3'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#0f3460'),
    ))
    
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading4'],
        fontSize=11,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor('#1a3a5c'),
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


def _parse_report_sections(report_text: str) -> dict:
    sections = {}
    
    header_match = re.search(
        r'Dataset:\s*(.+?)\n.*?'
        r'Timestamp:\s*(.+?)\n.*?'
        r'Target Column:\s*(.+?)\n',
        report_text, re.DOTALL
    )
    if header_match:
        sections['metadata'] = {
            'dataset': header_match.group(1).strip(),
            'timestamp': header_match.group(2).strip(),
            'target': header_match.group(3).strip(),
        }
    
    stage_pattern = r'(STAGE \d+(?:\.\d+)?:\s*[A-Z\s]+)\n-+\n'
    parts = re.split(stage_pattern, report_text)
    
    current_stage = None
    for i, part in enumerate(parts):
        if re.match(r'STAGE \d', part):
            current_stage = part.strip()
            sections[current_stage] = ""
        elif current_stage and part.strip():
            sections[current_stage] = part.strip()
    
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


def _format_json_to_flowables(json_text: str, styles) -> list:
    """Convert JSON content to professional formatted flowables with proper nesting."""
    flowables = []
    
    def format_value(value):
        """Format a single value for display."""
        if isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.4f}" if abs(value) < 1 else f"{value:,.2f}"
            return f"{value:,}"
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)
    
    def add_key_value(key, value, indent=0):
        """Add a key-value pair with proper formatting."""
        display_key = key.replace('_', ' ').title()
        escaped_key = _escape_xml(display_key)
        escaped_value = _escape_xml(format_value(value))
        
        style = styles['KeyValue'] if indent == 0 else styles['NestedBullet']
        prefix = "  " * indent
        
        flowables.append(Paragraph(
            f"{prefix}<b>{escaped_key}:</b> {escaped_value}",
            style
        ))
    
    def add_section_header(title, level=1):
        """Add a visual section header."""
        flowables.append(Spacer(1, 0.1*inch))
        style = styles['SectionHeader'] if level == 1 else styles['SubsectionHeader']
        flowables.append(Paragraph(
            _escape_xml(title.replace('_', ' ').title()),
            style
        ))
    
    def create_comparison_table(methods_data: dict):
        """Create a methods comparison summary table."""
        table_data = [['Method', 'Original', 'Result', 'Change', 'Imbalance Ratio']]
        
        for method_name, method_result in methods_data.items():
            if not isinstance(method_result, dict):
                continue
            
            mit = method_result.get('mitigation_result', {})
            comp = method_result.get('comparison_result', {})
            imb = comp.get('imbalance_metrics', {})
            
            orig_rows = mit.get('original_rows', 0)
            new_rows = mit.get('new_rows', orig_rows)
            change = new_rows - orig_rows
            
            orig_ratio = imb.get('original_imbalance_ratio', '-')
            mit_ratio = imb.get('mitigated_imbalance_ratio', '-')
            
            ratio_str = f"{orig_ratio:.2f} → {mit_ratio:.2f}" if isinstance(mit_ratio, (int, float)) else str(mit_ratio)
            
            table_data.append([
                method_name,
                f"{orig_rows:,}",
                f"{new_rows:,}",
                f"{change:+,}" if change != 0 else "0",
                ratio_str
            ])
        
        if len(table_data) > 1:
            table = Table(table_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.0*inch, 1.6*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            flowables.append(table)
            flowables.append(Spacer(1, 0.15*inch))
    
    def create_distribution_table(before: dict, after: dict):
        """Create a distribution comparison table."""
        table_data = [['Class', 'Before', 'After', 'Change']]
        
        for key in before.keys():
            b_val = before.get(key, 0)
            a_val = after.get(key, 0)
            change = a_val - b_val
            table_data.append([
                str(key),
                f"{b_val:,.0f}",
                f"{a_val:,.1f}" if isinstance(a_val, float) else f"{a_val:,}",
                f"{change:+,.1f}" if isinstance(change, float) else f"{change:+,}"
            ])
        
        if len(table_data) > 1:
            table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            flowables.append(table)
            flowables.append(Spacer(1, 0.1*inch))
    
    def process_method_result(method_name: str, method_data: dict):
        """Process a bias mitigation method result."""
        flowables.append(Spacer(1, 0.2*inch))
        flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e0e0e0')))
        flowables.append(Spacer(1, 0.1*inch))
        flowables.append(Paragraph(
            f"<b>{_escape_xml(method_name)}</b>",
            styles['SectionHeader']
        ))
        
        status = method_data.get('status', 'unknown')
        status_color = '#28a745' if status == 'success' else '#dc3545'
        flowables.append(Paragraph(
            f"Status: <font color='{status_color}'><b>{_escape_xml(status.upper())}</b></font>",
            styles['KeyValue']
        ))
        
        if 'mitigation_result' in method_data:
            mit = method_data['mitigation_result']
            add_section_header("Mitigation Summary", level=2)
            
            if 'method' in mit:
                flowables.append(Paragraph(
                    f"• <b>Technique:</b> {_escape_xml(mit['method'])}",
                    styles['BulletPoint']
                ))
            
            if 'original_rows' in mit and 'new_rows' in mit:
                orig = mit['original_rows']
                new = mit['new_rows']
                change = new - orig
                pct = (change / orig * 100) if orig > 0 else 0
                flowables.append(Paragraph(
                    f"• <b>Dataset Size:</b> {orig:,} → {new:,} ({pct:+.1f}%)",
                    styles['BulletPoint']
                ))
            
            if 'rows_added' in mit:
                flowables.append(Paragraph(
                    f"• <b>Samples Added:</b> +{mit['rows_added']:,}",
                    styles['BulletPoint']
                ))
            
            if 'weighted_imbalance_ratio' in mit:
                flowables.append(Paragraph(
                    f"• <b>Weighted Imbalance Ratio:</b> {mit['weighted_imbalance_ratio']:.2f}",
                    styles['BulletPoint']
                ))
            
            if 'k_neighbors' in mit:
                flowables.append(Paragraph(
                    f"• <b>K-Neighbors:</b> {mit['k_neighbors']}",
                    styles['BulletPoint']
                ))
            
            if 'distribution_before' in mit and 'distribution_after' in mit:
                add_section_header("Target Distribution", level=2)
                create_distribution_table(mit['distribution_before'], mit['distribution_after'])
            
            if 'sensitive_columns_used' in mit:
                cols = mit['sensitive_columns_used']
                flowables.append(Paragraph(
                    f"<b>Sensitive Attributes:</b> {_escape_xml(', '.join(cols))}",
                    styles['KeyValue']
                ))
            
            if 'note' in mit:
                flowables.append(Spacer(1, 0.05*inch))
                flowables.append(Paragraph(
                    f"<i>Note: {_escape_xml(mit['note'])}</i>",
                    styles['KeyValue']
                ))
        
        if 'comparison_result' in method_data:
            comp = method_data['comparison_result']
            
            if 'imbalance_metrics' in comp:
                imb = comp['imbalance_metrics']
                add_section_header("Imbalance Improvement", level=2)
                
                orig_ratio = imb.get('original_imbalance_ratio', 0)
                mit_ratio = imb.get('mitigated_imbalance_ratio', 0)
                improvement = imb.get('improvement', 'No')
                
                imp_color = '#28a745' if improvement == 'Yes' else '#dc3545'
                flowables.append(Paragraph(
                    f"• <b>Original Ratio:</b> {orig_ratio:.2f}",
                    styles['BulletPoint']
                ))
                flowables.append(Paragraph(
                    f"• <b>Mitigated Ratio:</b> {mit_ratio:.2f}",
                    styles['BulletPoint']
                ))
                flowables.append(Paragraph(
                    f"• <b>Improved:</b> <font color='{imp_color}'><b>{improvement}</b></font>",
                    styles['BulletPoint']
                ))
            
            if 'agent_analysis' in comp:
                add_section_header("Agent Analysis", level=2)
                analysis_text = comp['agent_analysis']
                clean_analysis = _clean_text_for_pdf(analysis_text)
                analysis_flowables = _format_markdown_to_paragraphs(clean_analysis, styles)
                flowables.extend(analysis_flowables)
    
    try:
        data = json.loads(json_text)
        
        if isinstance(data, dict) and 'methods' in data:
            status = data.get('status', 'unknown')
            status_color = '#28a745' if status == 'success' else '#dc3545'
            flowables.append(Paragraph(
                f"<b>Overall Status:</b> <font color='{status_color}'><b>{_escape_xml(status.upper())}</b></font>",
                styles['BodyText']
            ))
            
            if 'applied_methods' in data:
                methods_list = data['applied_methods']
                flowables.append(Paragraph(
                    f"<b>Methods Applied:</b> {_escape_xml(', '.join(methods_list))}",
                    styles['BodyText']
                ))
            
            if len(data['methods']) > 1:
                add_section_header("Methods Comparison", level=1)
                create_comparison_table(data['methods'])
            
            for method_name, method_data in data['methods'].items():
                if isinstance(method_data, dict):
                    process_method_result(method_name, method_data)
        
        elif isinstance(data, dict) and 'method' in data and 'original_rows' in data:
            status = data.get('status', 'unknown')
            status_color = '#28a745' if status == 'success' else '#dc3545'
            flowables.append(Paragraph(
                f"<b>Status:</b> <font color='{status_color}'><b>{_escape_xml(status.upper())}</b></font>",
                styles['KeyValue']
            ))
            
            if 'method' in data:
                flowables.append(Paragraph(
                    f"• <b>Technique:</b> {_escape_xml(data['method'])}",
                    styles['BulletPoint']
                ))
            
            if 'original_rows' in data and 'new_rows' in data:
                orig = data['original_rows']
                new = data['new_rows']
                change = new - orig
                pct = (change / orig * 100) if orig > 0 else 0
                flowables.append(Paragraph(
                    f"• <b>Dataset Size:</b> {orig:,} → {new:,} ({pct:+.1f}%)",
                    styles['BulletPoint']
                ))
            
            if 'rows_added' in data:
                flowables.append(Paragraph(
                    f"• <b>Samples Added:</b> +{data['rows_added']:,}",
                    styles['BulletPoint']
                ))
            
            if 'weighted_imbalance_ratio' in data:
                flowables.append(Paragraph(
                    f"• <b>Weighted Imbalance Ratio:</b> {data['weighted_imbalance_ratio']:.2f}",
                    styles['BulletPoint']
                ))
            
            if 'weight_statistics' in data:
                ws = data['weight_statistics']
                flowables.append(Paragraph(
                    f"• <b>Weight Range:</b> {ws.get('min', 0):.2f} - {ws.get('max', 0):.2f} (mean: {ws.get('mean', 0):.2f})",
                    styles['BulletPoint']
                ))
            
            if 'k_neighbors' in data:
                flowables.append(Paragraph(
                    f"• <b>K-Neighbors:</b> {data['k_neighbors']}",
                    styles['BulletPoint']
                ))
            
            if 'distribution_before' in data and 'distribution_after' in data:
                add_section_header("Target Distribution", level=2)
                create_distribution_table(data['distribution_before'], data['distribution_after'])
            
            if 'sensitive_columns_used' in data:
                cols = data['sensitive_columns_used']
                flowables.append(Paragraph(
                    f"<b>Sensitive Attributes:</b> {_escape_xml(', '.join(cols))}",
                    styles['KeyValue']
                ))
            
            if 'note' in data:
                flowables.append(Spacer(1, 0.05*inch))
                flowables.append(Paragraph(
                    f"<i>Note: {_escape_xml(data['note'])}</i>",
                    styles['KeyValue']
                ))
        
        elif isinstance(data, dict) and ('dataset_size' in data or 'imbalance_metrics' in data):
            status = data.get('status', 'unknown')
            status_color = '#28a745' if status == 'success' else '#dc3545'
            flowables.append(Paragraph(
                f"<b>Status:</b> <font color='{status_color}'><b>{_escape_xml(status.upper())}</b></font>",
                styles['KeyValue']
            ))
            
            if 'dataset_size' in data:
                ds = data['dataset_size']
                orig = ds.get('original', 0)
                mit = ds.get('mitigated', orig)
                diff = ds.get('difference', 0)
                pct = ds.get('percentage_change', 0)
                flowables.append(Paragraph(
                    f"• <b>Dataset Size:</b> {orig:,} → {mit:,} ({pct:+.1f}%)",
                    styles['BulletPoint']
                ))
            
            if 'target_distribution' in data:
                add_section_header("Target Distribution Changes", level=2)
                td = data['target_distribution']
                table_data = [['Class', 'Original', 'Mitigated', 'Change (pp)']]
                for class_name, class_data in td.items():
                    if isinstance(class_data, dict):
                        orig_pct = class_data.get('original_percentage', 0)
                        mit_pct = class_data.get('mitigated_weighted_percentage', 
                                    class_data.get('mitigated_percentage', 0))
                        pp_change = class_data.get('percentage_point_change', mit_pct - orig_pct)
                        table_data.append([
                            str(class_name),
                            f"{orig_pct:.1f}%",
                            f"{mit_pct:.1f}%",
                            f"{pp_change:+.1f}"
                        ])
                
                if len(table_data) > 1:
                    table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                    ]))
                    flowables.append(table)
                    flowables.append(Spacer(1, 0.1*inch))
            
            if 'imbalance_metrics' in data:
                add_section_header("Imbalance Improvement", level=2)
                imb = data['imbalance_metrics']
                orig_ratio = imb.get('original_imbalance_ratio', 0)
                mit_ratio = imb.get('mitigated_imbalance_ratio', 0)
                improvement = imb.get('improvement', 'No')
                
                imp_color = '#28a745' if improvement == 'Yes' else '#dc3545'
                flowables.append(Paragraph(
                    f"• <b>Original Ratio:</b> {orig_ratio:.2f}",
                    styles['BulletPoint']
                ))
                flowables.append(Paragraph(
                    f"• <b>Mitigated Ratio:</b> {mit_ratio:.2f}",
                    styles['BulletPoint']
                ))
                flowables.append(Paragraph(
                    f"• <b>Improved:</b> <font color='{imp_color}'><b>{improvement}</b></font>",
                    styles['BulletPoint']
                ))
                
                if 'note' in imb:
                    flowables.append(Paragraph(
                        f"<i>Note: {_escape_xml(imb['note'])}</i>",
                        styles['KeyValue']
                    ))
            
            if data.get('uses_weights'):
                flowables.append(Paragraph(
                    "<b>Note:</b> Results use sample weights for balanced training.",
                    styles['KeyValue']
                ))
        
        elif isinstance(data, dict):
            for key, value in data.items():
                if key in ('sensitive_attributes',):
                    flowables.append(Paragraph(
                        f"<b>{_escape_xml(key.replace('_', ' ').title())}:</b> <i>(detailed data omitted)</i>",
                        styles['KeyValue']
                    ))
                elif isinstance(value, dict) and len(value) > 10:
                    add_section_header(key)
                    flowables.append(Paragraph(
                        f"<i>Contains {len(value)} entries (details omitted for brevity)</i>",
                        styles['KeyValue']
                    ))
                elif isinstance(value, dict):
                    add_section_header(key)
                    for k, v in value.items():
                        if not isinstance(v, (dict, list)) or len(str(v)) < 200:
                            add_key_value(k, v, indent=1)
                elif isinstance(value, list):
                    add_key_value(key, value)
                else:
                    add_key_value(key, value)
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        add_key_value(key, value)
                    flowables.append(Spacer(1, 0.1*inch))
                else:
                    flowables.append(Paragraph(f"• {_escape_xml(str(item))}", styles['BulletPoint']))
    
    except json.JSONDecodeError:
        escaped = _escape_xml(json_text[:2000])
        flowables.append(Paragraph(escaped.replace('\n', '<br/>'), styles['Code']))
    except Exception:
        escaped = _escape_xml(json_text[:2000])
        flowables.append(Paragraph(escaped.replace('\n', '<br/>'), styles['Code']))
    
    return flowables


def _format_column_table(lines: list, styles) -> list:
    """Format 'Column: X | Reason: Y | Values: Z' lines as a proper table."""
    flowables = []
    table_data = [['Column', 'Category', 'Sample Values']]
    
    for line in lines:
        match = re.match(r'Column:\s*(.+?)\s*\|\s*Reason:\s*(.+?)\s*\|\s*Values:\s*\[(.+?)\]', line)
        if match:
            col_name = match.group(1).strip()
            reason = match.group(2).strip()
            values = match.group(3).strip()
            if len(values) > 40:
                values = values[:37] + "..."
            table_data.append([col_name, reason, values])
    
    if len(table_data) > 1:
        table = Table(table_data, colWidths=[2*inch, 2*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
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
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
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
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
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
        
        if re.match(r'Column:\s*.+?\s*\|\s*Reason:', stripped):
            flush_paragraph()
            column_lines.append(stripped)
            i += 1
            continue
        elif column_lines:
            flush_column_lines()
        
        if stripped.startswith('#### '):
            flush_paragraph()
            header_text = stripped[5:].replace('**', '')
            flowables.append(Paragraph(_escape_xml(header_text), styles['SubsectionHeader']))
            i += 1
            continue
        
        if stripped.startswith('### '):
            flush_paragraph()
            header_text = stripped[4:].replace('**', '')
            flowables.append(Paragraph(_escape_xml(header_text), styles['SectionHeader']))
            i += 1
            continue
        
        if stripped.startswith('## '):
            flush_paragraph()
            header_text = stripped[3:].replace('**', '')
            flowables.append(Paragraph(_escape_xml(header_text), styles['SectionHeader']))
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


def _format_bias_mitigation_section(text: str, styles) -> list:
    """Special formatter for Stage 6: Bias Mitigation that handles embedded JSON blocks."""
    flowables = []
    lines = text.split('\n')
    
    current_text = []
    in_json_block = False
    json_block_content = []
    json_brace_count = 0
    
    def flush_text():
        nonlocal current_text
        if current_text:
            text_content = '\n'.join(current_text)
            md_flowables = _format_markdown_to_paragraphs(text_content, styles)
            flowables.extend(md_flowables)
            current_text = []
    
    def flush_json():
        nonlocal json_block_content, json_brace_count, in_json_block
        if json_block_content:
            json_text = '\n'.join(json_block_content)
            json_flowables = _format_json_to_flowables(json_text, styles)
            flowables.extend(json_flowables)
            json_block_content = []
        json_brace_count = 0
        in_json_block = False
    
    for line in lines:
        stripped = line.strip()
        
        if not in_json_block and (stripped == '{' or stripped.startswith('{')):
            flush_text()
            in_json_block = True
            json_block_content = [line]
            json_brace_count = line.count('{') - line.count('}')
            continue
        
        if in_json_block:
            json_block_content.append(line)
            json_brace_count += line.count('{') - line.count('}')
            if json_brace_count <= 0:
                flush_json()
            continue
        
        current_text.append(line)
    
    flush_text()
    flush_json()
    
    return flowables


def generate_pdf_bytes(report_path: str) -> bytes:
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    sections = _parse_report_sections(report_text)
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
        flowables.append(Paragraph(f"Target Column: {meta.get('target', 'N/A')}", styles['Subtitle']))
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
        
        if 'BIAS MITIGATION' in section_name.upper():
            bias_flowables = _format_bias_mitigation_section(clean_content, styles)
            flowables.extend(bias_flowables)
        else:
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
