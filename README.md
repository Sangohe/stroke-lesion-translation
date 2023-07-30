# Lesion aware translation of ischemic stroke lesions

Gustavo Garzón, Santiago Gómez, Fabio Martínez

## Run

```bash
python run.py --config=configs/run.py:pix2pix,/home/sangohe/projects/lesion-aware-translation/data/APIS_synth-1_0_3_0_10_0_shuffled
```

# Examples

In the `examples` directory are stored the data and masks for patients with ischemic stroke lesions. Below is table with the details of each patient and the information about the lesion. **Note:** the slice index number is for the 1mm$^3$ resampled volumes, the numbers in parenthesis are for the original modalities.

| Patient ID | Difficulty | Slice index | Observations                                                                                            |
|------------|------------|-------------|---------------------------------------------------------------------------------------------------------|
| train_026  | Easy       | 45 (7)      | The NCCT presents an old lesion (easy difficulty) but the acute lesion is not visible (hard difficulty) |
| train_035  | Middle     | 45 (7)      | The NCCT presents a subtle hypoattenuation (middle difficulty)                                          |
| train_044  | Hard       | 99 (16)     | The lesion is clearly visible in the ADC image but not on NCCT (hard difficulty)                        |
| train_058  | Easy       | 104 (17)    | Control patient, the model should not generate any lesion (easy difficulty)                             |
| test_022   | Easy       | 69 (11)     | The lesion is clearly visible in the NCCT (easy difficulty)                                             |
| test_037   | Middle     | 105 (17)    | The lesion in the NCCT is located at a superior with a subtle hypoattentuation (middle difficulty)      |

# Todos

- [ ] Add more details about the run
- [ ] Add details about how to reproduce env