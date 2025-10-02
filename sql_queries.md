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

Example usage:
~~~~sql
SELECT
    col1, col2, col3
FROM
    f1;
~~~~

#### FROM

Purpose: Define what table to select data from

Example usage:
~~~~sql
SELECT
    *
FROM
    f1;
~~~~

#### AS

Purpose: Rename a column or table with an alias

Example usage:
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

Example usage:
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

...

### Subqueries

#### EXISTS

#### ALL

#### ANY

### Common Table Expressions

...

### Regular Expressions

...

## SQL Functions Review

### Aggregation Functions

...

### Window Functions

...

### String Functions

...

### Date Functions

...

### Math Functions

...


## References

[SQL Tutorial](https://www.sqltutorial.org)

[Geeks for Geeks SQL Tutorial](https://www.geeksforgeeks.org/sql/sql-tutorial/)


