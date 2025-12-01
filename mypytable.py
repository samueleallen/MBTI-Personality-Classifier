##############################################
# Description: Program contains various
# functions for data analytics, exploration,
# and cleaning.
##############################################
import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        N = len(self.data) # Get number of rows
        M = len(self.column_names) # Get number of columns

        return N, M

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            col: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        try:
            if isinstance(col_identifier, int):
                index = col_identifier
            elif isinstance(col_identifier, str):
                index = self.column_names.index(col_identifier)
            else:
                raise ValueError("Error: Column ID must be an int or string.")
        except (ValueError, IndexError):
            raise ValueError("Error: Invalid column id.")

        col_data = []

        # If include missing values is set to True, append all rows
        if include_missing_values:
            for row in self.data:
                value = row[index]
                col_data.append(value)
        # Else, only append instances where the value is not "NA"
        else:
            for row in self.data:
                value = row[index]
                if value != "NA":
                    col_data.append(value)

        return col_data
    
    def add_col(self, new_col_name, new_col_data):
        """
        Adds a new column to the table using a list of values
        
        Arguments:
            new_col_name (str): Name of new column to add
            new_col_data (list): List of values to be added under new column
        """
        # Add col_name to table
        self.column_names.append(new_col_name)

        # Loop through each instance, adding it to table data
        for i in range(len(self.data)):
            self.data[i].append(new_col_data[i])

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        # Loop through each row in table
        for i in range(len(self.data)):
            # Loop through each value in row
            for j in range(len(self.data[i])):
                value = self.data[i][j]
                try:
                    # Try to convert value to float
                    self.data[i][j] = float(value)
                except (ValueError, TypeError):
                    # If value cannot be converted to numeric, just move on to next value
                    pass

        return

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Sort (descending) list to make sure removing values doesn't mess up indices
        row_indexes_to_drop = sorted(row_indexes_to_drop, reverse=True)

        for index in row_indexes_to_drop:
            del self.data[index]

        return

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline='') as infile:
            reader = csv.reader(infile)

            # Read from first row to get columns
            self.column_names = next(reader)

            # Read rest of rows and store in data
            self.data = list(reader)

            # Try to convert rows to numeric
            self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='') as infile:
            writer = csv.writer(infile)

            # Write column names to first row
            writer.writerow(self.column_names)

            # Write rest of rows
            writer.writerows(self.data)

        return

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        key_indexes = []
        for col in key_column_names:
            key_indexes.append(self.column_names.index(col))

        # Create hashmap to store keys we have seen
        seen_instances = set()
        duplicate_indexes = []

        for i in range(len(self.data)):
            row = self.data[i]

            # Create list of values that make up key
            row_key_values = []
            for j in key_indexes:
                row_key_values.append(row[j])

            key = tuple(row_key_values)
            
            if key in seen_instances:
                duplicate_indexes.append(i)
            else:
                seen_instances.add(key)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        cleaned_table = []

        # Loop through each row
        for row in self.data:
            # If 'NA' not in row, append row to new, cleaned dataset
            if "NA" not in row:
                cleaned_table.append(row)

        self.data = cleaned_table

        return
                

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        # Calculate average of non NA values
        total = 0.0
        count = 0
        for row in self.data:
            value = row[col_index]
            if value != "NA":
                total += float(value)
                count += 1
        
        average = total / count

        # Fill previous missing values with average
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = average
        
        return

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        stats_cols = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []

        for col in col_names:
            # Get rid of any missing values
            clean_data = self.get_column(col, include_missing_values=False)

            # Handle edge case where there is no data
            if not clean_data:
                # Do nothing with the data and just move on
                continue

            col_min = min(clean_data)
            col_max = max(clean_data)
            col_mid = (col_min + col_max) / 2
            col_avg = sum(clean_data) / len(clean_data)

            # Calculate median
            sorted_data = sorted(clean_data)
            
            if len(clean_data) % 2 == 1:
                median = sorted_data[len(clean_data) // 2]
            else:
                mid1 = sorted_data[len(clean_data) // 2 - 1]
                mid2 = sorted_data[len(clean_data) // 2]
                median = (mid1 + mid2) / 2
        
            # Append stats list
            stats_data.append([col, col_min, col_max, col_mid, col_avg, median])


        return MyPyTable(stats_cols, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        self_key_indexes = []
        other_key_indexes = []
        for col in key_column_names:
            self_key_indexes.append(self.column_names.index(col))
            other_key_indexes.append(other_table.column_names.index(col))

        inner_join_data = []
        joined_cols = self.column_names.copy()

        # Add columns that aren't already in joined_cols from self.columns
        for i in range(len(other_table.column_names)):
            if i not in other_key_indexes:
                joined_cols.append(other_table.column_names[i])

        # Compare each row in both tables
        for row_self in self.data:
            for row_other in other_table.data:
                self_values = []
                # Store values from current table's row
                for i in self_key_indexes:
                    self_values.append(row_self[i])
                other_values = []
                # Store values from other table's row
                for i in other_key_indexes:
                    other_values.append(row_other[i])
                
                if self_values == other_values:
                    # Combine rows and skip any duplicate key columns
                    combined_row = row_self.copy()
                    for i in range(len(row_other)):
                        if i not in other_key_indexes:
                            combined_row.append(row_other[i])
                    inner_join_data.append(combined_row)

        return MyPyTable(joined_cols, inner_join_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        self_key_indexes = []
        other_key_indexes = []
        for col in key_column_names:
            self_key_indexes.append(self.column_names.index(col))
            other_key_indexes.append(other_table.column_names.index(col))

        outer_join_data = []
        joined_cols = self.column_names.copy()

        # Add columns that aren't already in joined_cols from self.columns
        for i in range(len(other_table.column_names)):
            if i not in other_key_indexes:
                joined_cols.append(other_table.column_names[i])

        # List to kepp track of rows matched with in other table
        matched_other_indexes = []

        # First perform join on tables from self
        for row_self in self.data:
            match = False
            
            for i in range(len(other_table.data)):
                row_other = other_table.data[i]

                # Compare key columns
                self_values = []
                for index in self_key_indexes:
                    self_values.append(row_self[index])

                other_values = []
                for index in other_key_indexes:
                    other_values.append(row_other[index])
                
                if self_values == other_values:
                    # Combine rows and skip columns from other_table
                    combined_row = row_self.copy()
                    
                    for j in range(len(row_other)):
                        if j not in other_key_indexes:
                            combined_row.append(row_other[j])
                    outer_join_data.append(combined_row)
                    matched_other_indexes.append(i)
                    match = True

            if not match:
                # Pad with NA values
                combined_row = row_self.copy()
                for j in range(len(other_table.column_names)):
                    if j not in other_key_indexes:
                        combined_row.append("NA")
                outer_join_data.append(combined_row)
        
        # Now add rows from other table that didnt have a match
        for i in range(len(other_table.data)):
            if i not in matched_other_indexes:
                row_other = other_table.data[i]

                combined_row = []

                # Now add rows from other table 
                for j in range(len(self.column_names)):
                    if j in self_key_indexes:
                        # Find foreign key index
                        key_index_in_other = other_key_indexes[self_key_indexes.index(j)]
                        combined_row.append(row_other[key_index_in_other])
                    else:
                        combined_row.append("NA")
                
                # Then add non-key columns from other table
                for j in range(len(row_other)):
                    if j not in other_key_indexes:
                        combined_row.append(row_other[j])
                outer_join_data.append(combined_row)

        return MyPyTable(joined_cols, outer_join_data)


    def drop_column(self, column_identifier):
        col_index = self.column_names.index(column_identifier)
        for i in range(len(self.data)):
            del self.data[i][col_index]
        del self.column_names[col_index]

    def even_class_distribution(self, column):
        counts = {}

        for element in column:
            if element in counts:
                counts[element] += 1
            else:
                counts[element] = 1

        frequent_label = max(counts, key=counts.get)
        non_frequent_label = min(counts, key=counts.get)
        to_delete = counts[frequent_label] - counts[non_frequent_label]

        new_column = []

        new_table = MyPyTable()
        new_table.column_names = self.column_names
        counter = 0
        i = 0
        while counter < counts[non_frequent_label] + 1:
            if column[i] == frequent_label:
                new_table.data.append(self.data[i])
                new_column.append(column[i])
                counter += 1
        for j in range(len(column)):
            if column[j] == non_frequent_label:
                new_table.data.append(self.data[j])
                new_column.append(column[j])
                    
                    

        
        return new_table, new_column

        


