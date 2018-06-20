import docx
from docx import Document
from openpyxl import Workbook


def get_input_xls_file_content(source_file):
    if not source_file:
        source_file = '/opt/techgig/MLTickets/docs/SampleInputDoc1-FAQs.docx'
    doc_obj = Document(source_file)
    paras = doc_obj.paragraphs
    return paras

def get_worksheet_obj():
    wb = Workbook()
    ws = wb.create_sheet("hardware")
    ws.title = "hardware_resolution"
    return [wb, ws]

def write_to_xls(source_file, dest_file):
    work_book, work_sheet = get_worksheet_obj()
    res_dict = {}
    title = None
    resolution = None
    row_count = 1
    paras = get_input_xls_file_content(source_file)
    #for row in range(1,len(paras)-50):
    title=None
    resolutions = []
    qna = {}
    tfs = 1
    pfs = 0
    tmpRes = None
    for row in range(0,len(paras)):
        para = paras[row]
        if para.style.style_id == "Heading4":
            print("======")
            print(para.text)
        """
        print("*** style: ", para.style.style_id)
        print("*** font:  ", para.style.font.size)
        print("*** bold:  ", para.style.font.bold)
        print("*** italic:", para.style.font.italic)
        for run in para.runs:
            if run.text == '\n':
                continue
            print("==========******====================")
            print(run.text)
            print("*** bold: ", run.bold)
            print("*** style: ", run.style.style_id)
            print("*** size: ", run.font.size)
            print("*** quick_style: ", run.style.quick_style)
        """

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide input docs file to convert into xlsx file")
    else:
        source_file = sys.argv[1]
        if sys.argv[2]:
            destination_file = sys.argv[2].strip()
        else:
            destination_file = None
        get_worksheet_obj()
        import pudb;pu.db
        write_to_xls(source_file, destination_file)
