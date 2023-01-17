from fpdf import FPDF
import os
root = '/work/shared/ptbc/CNN_Pancreas_V2/Donnees/EfficientNet/results/N_Tselect/pred_class_dist_png/'
images = os.listdir(root)[:5]
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(40, 10, 'CNN 1 : CLASS DISTRIBUTION ')
pdf.ln(20)
pdf.dashed_line(10, 30, 110, 30, 1, 10)
for img in images:
    pdf.image(f'{root}{img}', x = 50, y = None, w = 100, h = 70, type = '', link = '')

pdf.output('tuto1.pdf', 'F')
