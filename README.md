# F1 2025 WDC Predictor 🏎️📊

This project uses machine learning and real Formula 1 race data (including sprint points)  
to estimate the probability of each driver winning the **2025 World Drivers' Championship**  
with **2 races remaining** (Qatar & Abu Dhabi).

## What it does

- Fetches F1 race & sprint results using [FastF1](https://docs.fastf1.dev/)
- Rebuilds championship standings for modern seasons (2022–2025 era)
- Creates a "snapshot" of the points table with 2 races remaining
- Trains a logistic regression model to answer:
  > *Given the standings with 2 races left, who usually becomes champion?*
- Applies the model to the 2025 season after Round 22 and outputs WDC probabilities.

## Example 2025 Prediction (Model Output)

With 2 races remaining (after Round 22), the model outputs something like:

| Driver      | Points | Gap | WDC Probability |
|------------|--------|-----|-----------------|
| Lando Norris   | 390  | 0   | 99.9%          |
| Max Verstappen | 366  | 24  | 98.9%          |
| Oscar Piastri  | 366  | 24  | 98.9%          |
| George Russell | 294  | 96  | ~0.9%          |
| Others         | ...  | ... | ~0%            |

> Note: This is a statistical model, not a guarantee – it reflects patterns learned  
> from recent seasons and the size of the points gap with only two races remaining.

## Tech Stack

- Python
- FastF1
- pandas
- scikit-learn

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/ashwanths7528/f1-2025-wdc-predictor.git
   cd f1-2025-wdc-predictor
