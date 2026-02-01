# World Panel Dataset: Data Dictionary & Statistics

**File:** `world_panel_cleaned.csv`
**Dimensions:** 5984 rows × 23 columns
**Period:** 1990 - 2023
**Countries:** 176

## Variable Details

### Country Code
**Description:** ISO 3-letter country code

- **Type:** object
- **Unique Values:** 175

---

### Country Name
**Description:** Name of the country or region

- **Type:** object
- **Unique Values:** 176
- **Examples:** Albania, Algeria, Angola, Argentina, Armenia...

---

### Year
**Description:** Year of observation (1990-2023)

- **Type:** int64
- **Unique Values:** 34

---

### GDP_per_capita_constant_2015_US
**Description:** GDP per capita in constant 2015 US dollars. Indicator of average economic output per person.

| Statistic | Value |
| :--- | :--- |
| Count | 5962 |
| Mean | 12340.2293 |
| Std Dev | 17348.3151 |
| Min | 188.6583 |
| 25% | 1521.5357 |
| Median (50%) | 4409.4544 |
| 75% | 16240.4562 |
| Max | 112417.8770 |
| **Missing** | 22 (0.37%) |

---

### Trade_percent_of_GDP
**Description:** Sum of exports and imports of goods and services measured as a share of gross domestic product.

| Statistic | Value |
| :--- | :--- |
| Count | 5367 |
| Mean | 83.2588 |
| Std Dev | 51.3357 |
| Min | 0.0210 |
| 25% | 50.0447 |
| Median (50%) | 72.0438 |
| 75% | 101.3619 |
| Max | 442.6200 |
| **Missing** | 617 (10.31%) |

---

### Unemployment_total_percent_of_labor_force
**Description:** Unemployment refers to the share of the labor force that is without work but available for and seeking employment.

| Statistic | Value |
| :--- | :--- |
| Count | 5672 |
| Mean | 7.9250 |
| Std Dev | 5.9724 |
| Min | 0.1000 |
| 25% | 3.6160 |
| Median (50%) | 6.1990 |
| 75% | 10.6305 |
| Max | 38.8000 |
| **Missing** | 312 (5.21%) |

---

### Agriculture_value_added_percent_of_GDP
**Description:** Agriculture corresponds to ISIC divisions 1-5 and includes forestry, hunting, and fishing, as well as cultivation of crops and livestock production.

| Statistic | Value |
| :--- | :--- |
| Count | 5596 |
| Mean | 12.4121 |
| Std Dev | 11.4684 |
| Min | 0.0125 |
| 25% | 3.0581 |
| Median (50%) | 8.4986 |
| 75% | 19.2064 |
| Max | 64.6732 |
| **Missing** | 388 (6.48%) |

---

### Industry_value_added_percent_of_GDP
**Description:** Industry corresponds to ISIC divisions 10-45 and includes manufacturing (ISIC divisions 15-37).

| Statistic | Value |
| :--- | :--- |
| Count | 5577 |
| Mean | 27.4478 |
| Std Dev | 11.7763 |
| Min | 2.0863 |
| 25% | 19.9743 |
| Median (50%) | 25.5152 |
| 75% | 32.4244 |
| Max | 86.6696 |
| **Missing** | 407 (6.80%) |

---

### Urban_population_percent
**Description:** Urban population refers to people living in urban areas as defined by national statistical offices.

| Statistic | Value |
| :--- | :--- |
| Count | 5984 |
| Mean | 56.4082 |
| Std Dev | 23.0879 |
| Min | 5.2749 |
| 25% | 37.7011 |
| Median (50%) | 57.1286 |
| 75% | 74.7431 |
| Max | 100.0000 |
| **Missing** | 0 (0.00%) |

---

### Inflation_consumer_prices_annual_percent
**Description:** Inflation as measured by the consumer price index reflects the annual percentage change in the cost to the average consumer of acquiring a basket of goods and services.

| Statistic | Value |
| :--- | :--- |
| Count | 5445 |
| Mean | 25.6605 |
| Std Dev | 372.1108 |
| Min | -16.8597 |
| 25% | 1.9098 |
| Median (50%) | 4.1427 |
| 75% | 8.6423 |
| Max | 23773.1318 |
| **Missing** | 539 (9.01%) |

---

### Gross_fixed_capital_formation_percent_of_GDP
**Description:** Gross fixed capital formation (formerly gross domestic fixed investment) includes land improvements (fences, ditches, drains, and so on); plant, machinery, and equipment purchases; and the construction of roads, railways, and the like.

| Statistic | Value |
| :--- | :--- |
| Count | 5137 |
| Mean | 22.4109 |
| Std Dev | 7.6815 |
| Min | -2.4244 |
| 25% | 18.0300 |
| Median (50%) | 21.8498 |
| 75% | 25.8647 |
| Max | 93.5475 |
| **Missing** | 847 (14.15%) |

---

### Access_to_electricity_percent_of_population
**Description:** Access to electricity is the percentage of population with access to electricity.

| Statistic | Value |
| :--- | :--- |
| Count | 5355 |
| Mean | 80.3844 |
| Std Dev | 29.9030 |
| Min | 0.5339 |
| 25% | 68.4000 |
| Median (50%) | 99.0000 |
| 75% | 100.0000 |
| Max | 100.0000 |
| **Missing** | 629 (10.51%) |

---

### Internet_users_percent_of_population
**Description:** Internet users are individuals who have used the Internet (from any location) in the last 12 months. Internet can be used via a computer, mobile phone, personal digital assistant, games machine, digital TV etc.

| Statistic | Value |
| :--- | :--- |
| Count | 5287 |
| Mean | 30.5920 |
| Std Dev | 32.2134 |
| Min | 0.0000 |
| 25% | 1.5352 |
| Median (50%) | 16.8000 |
| 75% | 58.7413 |
| Max | 100.0000 |
| **Missing** | 697 (11.65%) |

---

### Services_value_added_percent_of_GDP
**Description:** Services correspond to ISIC divisions 50-99 and include value added in wholesale and retail trade (including hotels and restaurants), transport, and government, financial, professional, and personal services.

| Statistic | Value |
| :--- | :--- |
| Count | 5512 |
| Mean | 52.7344 |
| Std Dev | 12.2988 |
| Min | 6.4481 |
| 25% | 44.4534 |
| Median (50%) | 52.9635 |
| 75% | 60.8693 |
| Max | 96.1626 |
| **Missing** | 472 (7.89%) |

---

### Gross_domestic_savings_percent_of_GDP
**Description:** Gross domestic savings are calculated as GDP less final consumption expenditure (total consumption).

| Statistic | Value |
| :--- | :--- |
| Count | 5308 |
| Mean | 20.1259 |
| Std Dev | 17.5934 |
| Min | -136.8854 |
| 25% | 11.2418 |
| Median (50%) | 20.7267 |
| 75% | 29.1175 |
| Max | 87.8268 |
| **Missing** | 676 (11.30%) |

---

### Gov_expenditure_on_education_percent_of_GDP
**Description:** General government expenditure on education (current, capital, and transfers) is expressed as a percentage of GDP.

| Statistic | Value |
| :--- | :--- |
| Count | 3683 |
| Mean | 4.3675 |
| Std Dev | 1.9727 |
| Min | 0.0000 |
| 25% | 3.0802 |
| Median (50%) | 4.2079 |
| 75% | 5.3510 |
| Max | 44.3340 |
| **Missing** | 2301 (38.45%) |

---

### Primary_school_enrollment_gross_percent
**Description:** Gross enrollment ratio is the ratio of total enrollment, regardless of age, to the population of the age group that officially corresponds to the level of education shown.

| Statistic | Value |
| :--- | :--- |
| Count | 4431 |
| Mean | 99.7598 |
| Std Dev | 16.6015 |
| Min | 22.7073 |
| 25% | 95.6891 |
| Median (50%) | 101.1044 |
| 75% | 107.6770 |
| Max | 183.9894 |
| **Missing** | 1553 (25.95%) |

---

### RD_expenditure_percent_of_GDP
**Description:** Gross domestic expenditures on research and development (R&D), expressed as a percent of GDP. They include both capital and current expenditures.

| Statistic | Value |
| :--- | :--- |
| Count | 1191 |
| Mean | 1.0311 |
| Std Dev | 1.0473 |
| Min | 0.0102 |
| 25% | 0.2403 |
| Median (50%) | 0.6503 |
| 75% | 1.4556 |
| Max | 6.0192 |
| **Missing** | 4793 (80.10%) |

---

### Domestic_credit_to_private_sector_percent_of_GDP
**Description:** Domestic credit to private sector refers to financial resources provided to the private sector by financial corporations.

| Statistic | Value |
| :--- | :--- |
| Count | 4633 |
| Mean | 47.5727 |
| Std Dev | 43.9135 |
| Min | 0.3831 |
| 25% | 15.0610 |
| Median (50%) | 33.7216 |
| 75% | 64.5759 |
| Max | 301.0189 |
| **Missing** | 1351 (22.58%) |

---

### Fertility_rate_total
**Description:** Total fertility rate represents the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with age-specific fertility rates of the specified year.

| Statistic | Value |
| :--- | :--- |
| Count | 5984 |
| Mean | 3.0499 |
| Std Dev | 1.6307 |
| Min | 0.5860 |
| 25% | 1.7190 |
| Median (50%) | 2.5250 |
| 75% | 4.1753 |
| Max | 8.6060 |
| **Missing** | 0 (0.00%) |

---

### FDI_net_inflows_percent_of_GDP
**Description:** Foreign direct investment are the net inflows of investment to acquire a lasting management interest (10 percent or more of voting stock) in an enterprise operating in an economy other than that of the investor.

| Statistic | Value |
| :--- | :--- |
| Count | 5868 |
| Mean | 4.5827 |
| Std Dev | 19.6048 |
| Min | -391.5551 |
| 25% | 0.6787 |
| Median (50%) | 2.1861 |
| 75% | 4.7111 |
| Max | 452.2210 |
| **Missing** | 116 (1.94%) |

---

### Population_total
**Description:** Total population is based on the de facto definition of population, which counts all residents regardless of legal status or citizenship.

| Statistic | Value |
| :--- | :--- |
| Count | 5984 |
| Mean | 37249211.7154 |
| Std Dev | 136629781.1324 |
| Min | 40358.0000 |
| 25% | 2429235.5000 |
| Median (50%) | 7824847.0000 |
| 75% | 24148923.5000 |
| Max | 1438069596.0000 |
| **Missing** | 0 (0.00%) |

---

### Land_area_sq_km
**Description:** Land area is a country's total area, excluding area under inland water bodies, national claims to continental shelf, and exclusive economic zones.

| Statistic | Value |
| :--- | :--- |
| Count | 5863 |
| Mean | 718076.7727 |
| Std Dev | 1907817.1469 |
| Min | 20.0000 |
| 25% | 28020.0000 |
| Median (50%) | 143350.0000 |
| 75% | 527970.0000 |
| Max | 16389950.0000 |
| **Missing** | 121 (2.02%) |

---

