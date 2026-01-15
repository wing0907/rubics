import pandas as pd
import openpyxl

# Excel 파일 읽기
file_path = r"qube_out_readable/데이터분석_큐브_최종_장우진_readable_20260109_110903.xlsx"

try:
    # 시트 이름 확인
    xl_file = pd.ExcelFile(file_path)
    print("=== 시트 이름 ===")
    print(xl_file.sheet_names)
    print()
    
    # 첫 번째 시트 읽기
    df = pd.read_excel(file_path, sheet_name=0)
    print("=== 첫 번째 시트 정보 ===")
    print(f"행 수: {len(df)}")
    print(f"열 수: {len(df.columns)}")
    print(f"컬럼: {df.columns.tolist()}")
    print()
    print("=== 첫 3행 샘플 ===")
    print(df.head(3).to_string())
    
except Exception as e:
    print(f"오류: {e}")
