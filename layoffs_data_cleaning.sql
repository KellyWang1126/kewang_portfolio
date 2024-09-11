# Data cleaning

SELECT * 
FROM layoffs;


# Steps:
# 1. Remove duplicates
# 2. Standardize the data
# 3. Null values or blank values
# 4. Remove unnecessary columns


# Create a staging table; Never work with raw data directly
CREATE TABLE layoffs_staging
LIKE layoffs;

SELECT * 
FROM layoffs_staging;

INSERT layoffs_staging
SELECT * 
FROM layoffs;


# 1. Remove duplicates
# Identify duplicates
SELECT *, 
ROW_NUMBER() OVER (
PARTITION BY company, industry, total_laid_off, percentage_laid_off, 'date') AS row_num
FROM layoffs_staging;

WITH duplicate_cte AS
(
SELECT *, 
ROW_NUMBER() OVER (
PARTITION BY company, location, industry, total_laid_off, percentage_laid_off, `date`, stage, country, funds_raised_millions) AS row_num
FROM layoffs_staging
)
SELECT *
FROM duplicate_cte
WHERE row_num > 1;

# Take some examples for further check 
SELECT * 
FROM layoffs_staging
WHERE company = 'Casper';

# To actually remove the duplicates, create another staging table with row_num column
CREATE TABLE `layoffs_staging2` (
  `company` text,
  `location` text,
  `industry` text,
  `total_laid_off` int DEFAULT NULL,
  `percentage_laid_off` text,
  `date` text,
  `stage` text,
  `country` text,
  `funds_raised_millions` int DEFAULT NULL,
  `row_num` INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SELECT * 
FROM layoffs_staging2;

INSERT INTO layoffs_staging2
SELECT *, 
ROW_NUMBER() OVER (
PARTITION BY company, location, industry, total_laid_off, percentage_laid_off, `date`, stage, country, funds_raised_millions) AS row_num
FROM layoffs_staging;

# Check duplicates and delete them
SELECT * 
FROM layoffs_staging2
WHERE row_num > 1;

DELETE  
FROM layoffs_staging2
WHERE row_num > 1;

# Final check
SELECT * 
FROM layoffs_staging2;


# 2. Standardize the data
# Start with company
# Remove the empty space
SELECT company, TRIM(company)
FROM layoffs_staging2;

UPDATE layoffs_staging2
SET company = TRIM(company);

# Now industry
SELECT DISTINCT(industry)
FROM layoffs_staging2
ORDER BY 1;
 
# Some findings: missing values existed; duplicates existed such as Crypto, Crypto Currency and CryptoCurrency
SELECT *
FROM layoffs_staging2
Where industry LIKE 'Crypto%';

# Most named Crypto; So update others as Crypto
UPDATE layoffs_staging2
SET industry = 'Crypto'
WHERE industry LIKE 'Crypto%';

# Final check
SELECT DISTINCT(industry)
FROM layoffs_staging2
ORDER BY 1;

# Now location
SELECT DISTINCT(location)
FROM layoffs_staging2
ORDER BY 1;

# Now country
SELECT DISTINCT(country)
FROM layoffs_staging2
ORDER BY 1;

# United states has a period at the end
SELECT *
FROM layoffs_staging2
WHERE country LIKE 'United States%'
ORDER BY 1;

# Remove the period
SELECT DISTINCT(country), TRIM(TRAILING '.' FROM country)
FROM layoffs_staging2
ORDER BY 1;

UPDATE layoffs_staging2
SET country = TRIM(TRAILING '.' FROM country)
WHERE country LIKE 'United States%';

# Final check
SELECT DISTINCT(country)
FROM layoffs_staging2
ORDER BY 1;

# Now date
# Update date column from text to stardard date format in case we need to perform time series
SELECT `date`,
STR_TO_DATE(`date`, '%m/%d/%Y')
FROM layoffs_staging2;

UPDATE layoffs_staging2
SET `date` = STR_TO_DATE(`date`, '%m/%d/%Y');

# Change date type
ALTER TABLE layoffs_staging2
MODIFY COLUMN `date` DATE;

# Final check
SELECT *
FROM layoffs_staging2
ORDER BY 1;


# 3. Null values or blank values
SELECT *
FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

# Set empty space as null first
UPDATE layoffs_staging2
SET industry = NULL
WHERE industry = '';

SELECT *
FROM layoffs_staging2
WHERE industry IS NULL
OR industry = '';

# Populate data if applicable
SELECT *
FROM layoffs_staging2
WHERE company = 'Airbnb';

# Self join to find 
SELECT * 
FROM layoffs_staging2 t1
JOIN layoffs_staging2 t2
	ON t1.company = t2.company
WHERE t1.industry IS NULL
AND t2.industry IS NOT NULL; 

UPDATE layoffs_staging2 t1
JOIN layoffs_staging2 t2
	ON t1.company = t2.company
SET t1.industry = t2.industry
WHERE t1.industry IS NULL
AND t2.industry IS NOT NULL; 

# Check again 
SELECT company
FROM layoffs_staging2
WHERE industry IS NULL;

# Bally is the only industry with null now
SELECT *
FROM layoffs_staging2
WHERE company LIKE 'Bally%';


# 4. Remove unnecessary columns 
SELECT *
FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

DELETE 
FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

ALTER TABLE layoffs_staging2
DROP COLUMN row_num;

# Final check
SELECT *
FROM layoffs_staging2;

