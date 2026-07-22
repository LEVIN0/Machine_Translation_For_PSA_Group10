from openpyxl import load_workbook
from pathlib import Path
from datetime import datetime


class ExcelManager:

    def __init__(self):

        self.file = Path("data/external/PSA_Parallel_Dataset.xlsx")

        self.workbook = load_workbook(self.file)

        self.sheet = self.workbook["Master_Dataset"]

    def add_record(
        self,
        psa_id,
        domain,
        english,
        source,
        metadata="scraped"
    ):

        self.sheet.append([
            psa_id,
            domain,
            english,
            "",          # Kiswahili
            "",          # Dholuo
            source,
            datetime.today().strftime("%Y-%m-%d"),
            metadata,
            "Pending"
        ])

    def save(self):

        self.workbook.save(self.file)