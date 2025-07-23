
import requests
import bz2
import re
import pandas as pd


def download_and_uncompress_bz_to_string(url: str) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        bz_content = response.content
    decompressed_content = bz2.decompress(bz_content)
    return decompressed_content.decode('latin-1')

def download_txt_file_to_string(url: str) -> str:
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        content = response.text
    return content



# Define a function to extract the metadata and column structure from the text
def parse_column_names(file_content: str) -> [str]:
    column_info = []
    column_pattern = re.compile(r"^Column[s]?\s(\d+)([-\d]*)")
    for line in file_content.splitlines():
        match = column_pattern.match(line)
        if match:
            column_range = match.group(2)
            if not column_range:
                column_info.append(line.split(":")[1].strip())
            else:
                start_col = int(match.group(1))
                end_col = int(column_range[1:])
                for i in range(start_col, end_col + 1):
                    column_info.append(f"Mean over all cycles {i - start_col + 1}")
                break  # we can break here we skip the rest of the file
    return column_info


def get_data_lines(file_content: str) -> [str]:
    # We are looking for the SQ lines (but I forgot what it stands for, Sun Quality?)
    pattern = re.compile(r"SQ\s(\d{8}T\d{6})")
    lines = file_content.splitlines()
    data_lines = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            data_lines.append(line)
    return data_lines


def df_from_string(file_content: str) -> pd.DataFrame | None:
    columns = parse_column_names(file_content)
    data_lines = get_data_lines(file_content)
    if len(data_lines) > 0:
        data = [line.split()[:len(columns)] for line in data_lines]
        df = pd.DataFrame(data, columns=columns)
        return df
    return None


def get_filelinks_in_folder(url: str, start_date: str, end_date: str) -> [str]:
    response = requests.get(url)
    response.raise_for_status()
    files = re.findall(r'href=[\'"]?([^\'" >]+)', response.text)
    files = [f for f in files if (f.endswith('.bz2') or f.endswith('.txt'))]

    if "/L1/" in url:
        files = filter_for_newest_files(files)

    date_pattern = re.compile(r"20\d{6}")
    dates = []
    for file in files:
        match = date_pattern.search(file)
        if match:
            dates.append(match.group())
    files = [f for f, d in zip(files, dates) if start_date <= d <= end_date]
    return files

def filter_for_newest_files(urls: [str]) -> [str]:
    """
    For L1 files we can have multiple versions of the same date, so we need to check the version number and the valid date.
    """
    dates = []
    cversions = []
    valids = []
    columns = ["urls", "date", "cversion",  "valid"]
    for url in urls:
        date = re.search(r"\d{8}", url).group()
        cversion = re.search(r"smca1c(\d+)", url).group(1)
        valid = re.search(r"20\d{6}.*(\d{8})", url).group(1)
        dates.append(date)
        cversions.append(cversion)
        valids.append(valid)
    df = pd.DataFrame(list(zip(urls, dates, cversions, valids)), columns=columns)
    # sort by date, valid and cversion
    df = df.sort_values(["date", "valid", "cversion"], ascending=[False, False, False])
    df = df.drop_duplicates(subset="date", keep="first")
    return df["urls"].tolist()





def download_data_files(url: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download all files from the given URL that are within the date range.
    Dates are in the format YYYYMMDD
    """
    file_links = get_filelinks_in_folder(url, start_date, end_date)
    print(f"Downloading {len(file_links)} files")
    df = pd.DataFrame()
    i = 1
    for link in file_links:
        try:
            print(f"Downloading file {i} of {len(file_links)} with link {link}")
            file_url = url + link
            if (file_url.endswith(".txt")):
                file_string = download_txt_file_to_string(file_url)
            else:
                file_string = download_and_uncompress_bz_to_string(file_url)
            file_df = df_from_string(file_string)
            if file_df is not None:
                df = pd.concat([df, file_df])
        except Exception as e:
            print(f"Failed to download {link} with error {e}")
        i += 1
    return df
