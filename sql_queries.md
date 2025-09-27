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
    *
FROM
    f1
~~~~

#### FROM

...

#### AS

...

### Data Ordering

#### ORDER BY

...

#### ASC/DESC

...

### Limiting Rows

#### DISTINCT

#### LIMIT

### Filtering Rows

#### WHERE

#### AND/OR

#### BETWEEN

#### IN

#### LIKE

#### IS NULL

#### NOT

### Joining Tables

...

### Grouping Rows

...

### Aggregation Functions

...

### Set Operations

...

### Conditionals

...

### Subqueries

#### EXISTS

#### ALL

#### ANY

### Common Table Expressions

...

### Window Functions

...

### Regular Expressions

...

### String Manipulation

...

### Date Manipulation

...


## References

[SQL Tutorial](https://www.sqltutorial.org)


