# SQL Queries Quick Review

SQL is a high-level, declarative data transformation language supported by relational databases and other data manipulation software like spark. SQL consists of multiple sublanguages: data query language, used for performing queries on data, data definition language, used for creating and modifying objects in databases, data control language, used to control access to data, and data manipulation language, used for inserting, deleting and modifying data.

ANSI SQL provides standardized version of SQL. Databases generally don't fully adhere to this standardized version though, often requiring some changes when changing databases. 

## Some Example SQL Queries

SQL queries consist of series of clauses. Some examples include:

~~~~sql
SELECT
    s1, s2, ...
FROM
    f1
JOIN
    f2 on join_condition
GROUP BY
    g1
HAVING
    h1, h2
LIMIT
    l1;
~~~~

...

## SQL Keywords Review

Many of the most widely used keywords are reviewed in the below table

### Data Selection

#### SELECT

Purpose: Define what columns/calculated columns to select

Example:
~~~~sql
SELECT
    col1, col2, col3
FROM
    f1;
~~~~

#### FROM

Purpose: Define what table to select data from

Example:
~~~~sql
SELECT
    *
FROM
    f1;
~~~~

#### AS

Purpose: Rename a column or table with an alias

Example:
~~~~sql
SELECT
    col1 AS renamed_col
FROM
    f1;
~~~~

### Data Ordering

#### ORDER BY/ASC/DESC

Purpose: Sort the result set in ascending or descending order.

The result set is sorted in ascending order by default, which can be more explicitly stated by including the ASC keyword or changed to descending order by including the DESC keyword.

Example:
~~~~sql
SELECT
    *
FROM
    f1
ORDER BY
    name DESC;
~~~~

### Limiting Rows

#### DISTINCT

Purpose: Return only distinct values in the result set

Example usage:
~~~~sql
SELECT DISTINCT
    country
FROM
    customers;
~~~~

#### LIMIT

Purpose: Specify number of records to return

Example usage:
~~~~sql
SELECT
    *
FROM
    f1
LIMIT
    3;
~~~~

### Filtering Rows

#### WHERE

Purpose: Filter the result set to only include records that fulfill a condition.

Example usage:
~~~~sql
SELECT
    *
FROM
    Customers
WHERE
    Country='USA';
~~~~

#### AND/OR

Purpose: Used to combine conditions.

Example usage:
~~~~sql
SELECT
    *
FROM
    Customers
WHERE
    Country='USA' OR Country='MEXICO';
~~~~

#### BETWEEN

Purpose: Create conditions for values being between points. The values can be numbers, text or dates.

Example:
~~~~sql
SELECT *
FROM orders
WHERE OrderDate BETWEEN '1996-07-01' AND '1996-07-31';
~~~~

#### IN

Purpose: Create conditions for values in a set of values.

Example:
~~~~sql
SELECT *
FROM Customers
WHERE Country in ('USA', 'Mexico');
~~~~

#### LIKE

The LIKE operator returns True if a value matches a pattern and false otherwise. The LIKE operator has two wildcard characters: '%' matches zero, one or more characters, and '_' matches exactly one character.

Example:


#### IS NULL

#### NOT

### Joining Tables

Join keywords are used to merge rows of tables based on conditions. The different join keywords specify different conditions for when to merge rows. 

| Keyword | Join Condition |
|--------|-----------------|
| INNER JOIN/JOIN | Merges rows where condition is true |
| LEFT JOIN | Merge rows where condition is true and leftover rows from first table with null values for columns from second table |
| RIGHT JOIN | Merge rows where condition is ture and leftover rows from second table with null values for columns from first table |
| CROSS JOIN | Combine all rows of left table with all rows of right table (does not require a condition) |

Example:
~~~~sql
SELECT table1.col1, table2.col2
FROM table1
INNER JOIN table2 ON table1.col1=table2.col1;
~~~~

### Grouping Rows

#### GROUP BY

Purpose: 

#### HAVING

#### GROUPING SETS

#### ROLLUP

### Set Operations

#### UNION/UNION ALL

...

#### INTERSECT

...

#### MINUS

...

### Conditionals

#### CASE/WHEN/THEN/ELSE/END

Purpose: Add conditional logic to SQL queries.

Example:
~~~~sql
SELECT
    product_name,
    CASE
        WHEN price < 10 THEN 'cheap'
        WHEN price >= 10 AND price <= 20 THEN 'average'
        WHEN price >= 20 THEN 'expensive'
        ELSE 'unknown'
    END price_category
FROM
    products;
~~~~

#### COALESCE

Purpose: Takes one or more arguments and returns first non-null argument.

Example:
~~~~sql
SELECT COALESCE(NULL, 'non-null') AS example;
~~~~

### Subqueries

A SQL subquery is a query nested within another query. A correlated subquery is a special type of subquery which references a column from the outer query. Correlated subqueries, unlike normal subqueries, are executed once for each row from the outer query.

Examples:
...

#### EXISTS

Purpose: Used to check if a value exists within the results returned by a subquery.

Example:
...

#### ALL

Purpose: used to compare a value to all values returned by a subquery. Returns true if the specified condition holds for all values returned by the subquery.

Example:
...

#### ANY

Purpose: used to compare a value to all values returned by a subquery. Returns true if the specified condition holds for any values returned by the subquery.

Example:
...

### Regular Expressions

...

### Common Table Expressions

#### WITH

Purpose: Define a temporary result set that can be referenced within a query

Example:
~~~~sql
WITH MonthlySales AS (
    SELECT
        DATE_TRUNC('month', order_date) AS sales_month,
        SUM(amount) AS total_monthly_sales
    FROM
        orders
    GROUP BY
        sales_month
)
SELECT
    sales_month,
    total_monthly_sales
FROM
    MonthlySales
WHERE
    total_monthly_sales > 10000;
~~~~

### Other

#### INTERVAL

Purpose: Define a timespan such as years, days or seconds.

Example:
~~~~sql
SELECT date + INTERVAL '1 month';
~~~~

## SQL Functions Review

### Aggregation Functions

SQL aggregate functions calculate on a set of values returning a single value. Some of the most common aggregate functions are

| Function | Purpose |
| -------- | ------- |
| AVG | Average of set |
| ANY_VALUE | Any value in set |
| COUNT | Number of items in set |
| MAX | Maximum value in set |
| MIN | Minimum value in set |
| SUM | Sum of values in set |

### Window Functions

In addition to the aggregation functions, there are also specific ranking and value window functions that depend on the ordering of items in a window. Some of the most common ranking window functions are

| Function | Purpose |
| -------- | ------- |
| RANK | ... |
| DENSE_RANK | ... |
| ROW_NUMBER | ... |
| PERCENT_RANK | ... |
| NTILE | ... |

Some of the most common value window functions are

| Function | Purpose |
| -------- | ------- |
| LAG(column, offset, default) | Retrieve value of column from previous row within the partition |
| LEAD(column, offest, default) | Retrieve value of column from a succeeding row within the partition |
| FIRST_VALUE | Return value of column for the first row in the partition |
| LAST_VALUE | Return value of column for the last row in the partition |

### String Functions

SQL string functions can be applied to string columns and values for processing. Some of the most common string functions are

| Function | Purpose |
| -------- | -------- |
| CONCAT | Concatenate multiple strings |
| UPPER/LOWER | Convert text to upper/lower case |
| LENGTH | Length of string in characters |
| SUBSTRING(string, start, length) | Extracts a substring of string |
| REPLACE(string, old_substring, new_substring) | Replace occurrences of a substring |
| INSTR(string, substring) | Find the position of the first occurrence of a substring within a string |

### Date Functions

SQL date functions can be applied to date columns and values for processing. Some of the most common date functions are

| Function | Purpose |
| -------- | ------- |
| CURRENT_DATE | |
| CURRENT_TIME | |
| NOW | |
| EXTRACT | Extract specific date and time components |
| TO_DATE | Convert string to date format |
| TO_CHAR | Convert date or timestamp into formatted string |

### Math Functions

SQL math functions can be applied to columns and values for processing. Some of the most common are

| Function | Purpose |
| -------- | ------- |
| ABS      | Absolute value |
| FLOOR | Round a number down to the nearest integer |
| LOG | Base 10 logarithm |
| POWER | Raise a number to the specified power |
| ROUND | Round a number |
| RANDOM | Generate a random number between 0 and 1 |




## References

[SQL Tutorial](https://www.sqltutorial.org)

[Geeks for Geeks SQL Tutorial](https://www.geeksforgeeks.org/sql/sql-tutorial/)


