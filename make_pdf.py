"""Build the submission PDF (links, README content, screenshot) from README.md."""
import re, os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, PageBreak

GITHUB = sys.argv[1] if len(sys.argv) > 1 else "GITHUB_LINK_HERE"
APP = sys.argv[2] if len(sys.argv) > 2 else "STREAMLIT_LINK_HERE"
OUT = sys.argv[3] if len(sys.argv) > 3 else "ML_Assignment_2_Submission.pdf"
SHOTS = sys.argv[4:]
S = getSampleStyleSheet()
cell = S["BodyText"].clone("cell", fontSize=8, leading=10)

def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)

md = open("README.md", encoding="utf-8").read().replace("GITHUB_LINK_HERE", GITHUB).replace("STREAMLIT_LINK_HERE", APP)
st = [Paragraph("Machine Learning - Assignment 2", S["Title"]),
      Paragraph("M.Tech (AIML), WILP", S["Normal"]), Spacer(1, 10),
      Paragraph("1. GitHub Repository Link", S["Heading2"]), Paragraph(inline(GITHUB), S["Normal"]),
      Paragraph("2. Live Streamlit App Link", S["Heading2"]), Paragraph(inline(APP), S["Normal"]),
      Paragraph("3. BITS Virtual Lab Screenshot", S["Heading2"])]
if SHOTS:
    for n, p in enumerate(SHOTS, 1):
        st += [Paragraph(f"Screenshot {n} of {len(SHOTS)}", S["Normal"]),
               Image(p, width=480, height=270, kind="proportional"), Spacer(1, 10)]
else:
    st.append(Paragraph("[Insert BITS Virtual Lab screenshot here]", S["Normal"]))
st += [PageBreak(), Paragraph("4. README.md content", S["Heading2"])]

lines, i = md.splitlines(), 0
while i < len(lines):
    l = lines[i]
    if l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.match(r"^\|[-| ]+\|$", lines[i]):
                rows.append([Paragraph(inline(c.strip()), cell) for c in lines[i].strip("|").split("|")])
            i += 1
        n = len(rows[0])
        w = [110] + [(420 - 110) / (n - 1)] * (n - 1) if n > 2 else [110, 310]
        t = Table(rows, colWidths=w, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        st += [t, Spacer(1, 8)]; continue
    if l.startswith("```"):
        j = i + 1; code = []
        while not lines[j].startswith("```"): code.append(lines[j]); j += 1
        st.append(Preformatted("\n".join(code), S["Code"])); i = j + 1; continue
    if l.startswith("### "): st.append(Paragraph(inline(l[4:]), S["Heading3"]))
    elif l.startswith("## "): st.append(Paragraph(inline(l[3:]), S["Heading2"]))
    elif l.startswith("# "): st.append(Paragraph(inline(l[2:]), S["Heading1"]))
    elif l.startswith("- "): st.append(Paragraph("&bull; " + inline(l[2:]), S["Normal"]))
    elif l.strip(): st.append(Paragraph(inline(l), S["Normal"]))
    i += 1
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=50, rightMargin=50).build(st)
print("built")
