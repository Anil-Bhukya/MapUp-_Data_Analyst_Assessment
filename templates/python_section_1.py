from typing import Dict, List
import pandas as pd
import re
import itertools
from math import radians, sin, cos, sqrt, atan2


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = [] 
    for i in range(0, len(lst), n): 
        group = lst[i:i + n] 
        group.reverse() 
        result.extend(group) 
    return result 

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {} 
    
    for string in lst: 
        length = len(string) 
        
        if length not in length_dict: 
            length_dict[length] = [] 
            
        length_dict[length].append(string)  
    
    return length_dict 

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten(current_dict, parent_key=""):
        items = []
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, elem in enumerate(v):
                    if isinstance(elem, dict):
                        items.extend(flatten(elem, f"{new_key}[{i}]").items())
                    else:
                        items.append((f"{new_key}[{i}]", elem))
            else:
                items.append((new_key, v))
        return dict(items)
    
    return flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return [list(p) for p in set(itertools.permutations(nums))]


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\d{2}-\d{2}-\d{4}',    # dd-mm-yyyy
        r'\d{2}/\d{2}/\d{4}',    # mm/dd/yyyy
        r'\d{4}\.\d{2}\.\d{2}'   # yyyy.mm.dd
    ]
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    import polyline
    
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of the Earth in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2]) # Convert degrees to radians
        # Differences in coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        # Haversine formula
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c  # Distance in meters
        return distance
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)    
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
        n = len(matrix)
        rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)] #rotating matrix by 90 degrees
        
        # Calculating row sums and column sums
        row_sums = [sum(row) for row in rotated]
        col_sums = [sum(rotated[i][j] for i in range(n)) for j in range(n)]
        
        # Creating the transformed matrix
        transformed = [
            [row_sums[i] + col_sums[j] - 2 * rotated[i][j] for j in range(n)]
            for i in range(n)
        ]
    
        return transformed



    def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify if each unique (id, id_2) pair covers a full 24-hour period and spans all 7 days of the week.
    
    Args:
        df (pandas.DataFrame)
    
    Returns:
        pd.Series: A boolean series indicating if the (id, id_2) pair has correct timestamps.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.day_name()  # Day of the week (Monday, Tuesday, etc.)
    df['time'] = df['timestamp'].dt.time  # Extract the time part
    
    # Function to check if the pair covers both all days and the full 24-hour period
    def is_full_coverage(sub_df):
        # Step 1: Check for 7-day coverage
        days = set(sub_df['day'])
        if len(days) < 7:
            return False
        
        # Step 2: Check for 24-hour coverage (00:00:00 to 23:59:59)
        for day in days:
            day_times = sorted(sub_df[sub_df['day'] == day]['time'])
            # Check if the earliest time is 00:00:00 and the latest time is 23:59:59
            if day_times[0] != pd.to_datetime('00:00:00').time() or day_times[-1] != pd.to_datetime('23:59:59').time():
                return False
        
        # If both conditions are met, return True
        return True
    
    # Apply the check for each (id, id_2) group
    result = df.groupby(['id', 'id_2']).apply(is_full_coverage)
    
    return result
