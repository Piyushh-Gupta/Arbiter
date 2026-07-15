import zipfile, xml.etree.ElementTree as ET, sys
def extract_text(docx_path, out_path):
    z = zipfile.ZipFile(docx_path)
    root = ET.fromstring(z.read('word/document.xml'))
    text = '\n'.join(node.text for node in root.iter() if node.text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

extract_text('Arbiter_SRD_Final.docx', 'SRD.txt')
extract_text('Arbiter_SDD_Final.docx', 'SDD.txt')
extract_text('Arbiter_MEEP_Final.docx', 'MEEP.txt')
