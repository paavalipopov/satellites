# Material reflectance sheets: anomalies to check

Prepared 2026-10-23 for Elena. Data: `Bus.xlsx`, `Wings.xlsx`, `antenna.xlsx`, 46 materials (35 with an irradiated spectrum). For the synthetic satellite images, each spectrum is linearly interpolated onto 36 bands over 350–1000 nm (one every 18.6 nm).

## Current handling

1. Grid wavelengths outside a spectrum's scan take the nearest measured value: the first one at the blue end, the last one at the red end.
2. All values are clipped to [0, 1].

On our grid, starting at 390 nm instead of 350 nm would leave no value beyond [−0.05, 1.05].

## Anomalies within 350–1000 nm

Values as they appear in the sheets (P = pristine, I = irradiated).

| material | state | issue | wavelengths | extreme values |
|---|---|---|---|---|
| AZW-LAII | P / I | far outside [0, 1] at the start of the scan | 350–351 / 350–368 nm | +4.45 / +15.5 |
| AZ-2000 | P / I | same | 351–360 / 350–377 nm | −0.66 / +7.72 |
| AZ-400 | P / I | same | 350–373 / 350–354 nm | −0.57, +1.75 / −8.30, +1.38 |
| AZJ-4020 | P / I | same | 351–364 / 350–377 nm | −0.98 / −9.81, +4.99 |
| AZ-3700 | P / I | same | 350–361 / 350–387 nm | −0.50, +1.29 / −5.13, +4.13 |
| AZ-2100 | P / I | same | 350–378 / 350–364 nm | −1.82, +2.62 / +5.15 |
| AZ-93 | P / I | same | 350–351 / 350–358 nm | +7.21 / +4.27 |
| Black paint | P / I | negative at the start of the scan | 368–379 / 351–366 nm | −0.02 / −0.37 |
| Aluminum | P | above 1 at the start of the scan | 351–394 nm | +1.14 |
| Aluminum 6061 | P | slightly above 1 | 542–631 nm | +1.007 |
| Solar_cell | P / I | slightly negative | 885–982 / 854 nm | −0.008 / just below 0 |
| Mylar_aluminized | P, I | scan starts at 430.5 nm | 350–430 nm take the 430.5 nm value | — |
| BetaCloth | P | scan ends at 999 nm | the 1000 nm band takes the 999 nm value | — |
| Gold_foil | P | far too dark for gold | whole scan | 0.2–0.6 % everywhere; polished gold reflects about 40 % in the blue and over 90 % above 600 nm |
| Copper_foil | P | too dark, and decreasing into the infrared | whole scan | 6 % at 350 nm, peak 26 % at 700 nm, 12 % at 2400 nm; the `Copper` sheet gives 16–84 %, rising |
| GFRP (antenna), Glass_Fiber (Wings) | P, I | identical data in two sheets | whole scan | — |

Small gaps at the blue end are filled the same way and are negligible: Ge solar cell starts at 355.4 nm, Aluminum 6061 at 353.2 nm, Stainless_Steel_2 at 352.3 nm, Stainless_steel_1, BetaCloth and Gold_foil at 351 nm.

## Beyond 1000 nm (not used yet; matters if the grid is extended)

- Most spectra leave [0, 1] somewhere beyond 1400–2400 nm; Aluminum, Aluminum 6061 and CMG_with_Cu already from about 1000–1100 nm.
- Mylar_aluminized ends at 2198 nm.
- Most sheets have scattered missing values from about 974 nm on.
- Si solar cell repeats 10 wavelengths between 1034 and 1097 nm, with different values.

## Questions

1. Is filling with the nearest measured value acceptable at the scan ends, or should those bands be dropped for these materials?
2. Are the blue-end values of the AZ coatings, Black paint and Aluminum measurement artifacts? Is clipping to [0, 1] reasonable, or should the grid start around 390 nm?
3. Gold_foil and Copper_foil: were they measured differently (for example diffuse only, specular excluded), or are the units different? Is a total-reflectance measurement available?
4. GFRP and Glass_Fiber: the same material, or was one sheet copied by mistake?
5. Eleven materials have no irradiated spectrum (Stainless_steel_1, Stainless_Steel_2, BetaCloth, Aluminum 7075T6, Aluminum 6061, Si solar cell, Ge solar cell, Titanium, Copper_foil, Copper, Gold_foil): not measured, or missing from the files?
6. Six spectra are sparse with irregular wavelengths (for example 342.033 nm), as if digitized from a figure: Aluminum 7075T6, Aluminum 6061, Si solar cell, Ge solar cell, Titanium, Copper. What is their source?
7. Minor: steps of about 0.03 near 400–416 nm (Thermalbright_N, Kapton_CS), Kapton-WS-glossy) and 975–990 nm (CORIN_XLS, Solar_cell). Instrument changeover?
