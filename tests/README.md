# Demo images

Curated leaf photographs for demonstrating the system. Every image is from
the **test split**, so none of them was seen during training.

For each image the table gives a farm and date that actually grow that crop
in its real season - pair them as shown and the forecast will be sensible.

## How to use

1. `python scripts/06_serve.py`
2. Open <http://127.0.0.1:8000> and go to **Live System**
3. Drop in an image below, set the suggested farm and date, run the pipeline

## Best for demonstration

| File | True class | Farm | Date | Model confidence |
|---|---|---|---|---|
| `01_tomato_late_blight.JPG` | Tomato — Late blight | Kolar, Karnataka | 2023-07-15 | 96.4% |
| `02_tomato_early_blight.JPG` | Tomato — Early blight | Kolar, Karnataka | 2023-07-15 | 93.3% |
| `03_tomato_spider_mites_two-spotted_spider_mite.JPG` | Tomato — Spider mites Two-spotted spider mite | Kolar, Karnataka | 2023-07-15 | 95.0% |
| `04_tomato_tomato_yellow_leaf_curl_virus.JPG` | Tomato — Tomato Yellow Leaf Curl Virus | Kolar, Karnataka | 2023-07-15 | 97.9% |
| `05_tomato_septoria_leaf_spot.JPG` | Tomato — Septoria leaf spot | Kolar, Karnataka | 2023-07-15 | 96.2% |
| `06_tomato_healthy.JPG` | Tomato — healthy | Kolar, Karnataka | 2023-07-15 | 96.2% |
| `07_potato_late_blight.JPG` | Potato — Late blight | Agra, Uttar Pradesh | 2023-11-20 | 94.5% |
| `08_potato_early_blight.JPG` | Potato — Early blight | Agra, Uttar Pradesh | 2023-11-20 | 96.9% |
| `10_apple_apple_scab.JPG` | Apple — Apple scab | Shimla, Himachal Pradesh | 2023-06-15 | 86.3% |
| `11_apple_black_rot.JPG` | Apple — Black rot | Shimla, Himachal Pradesh | 2023-06-15 | 86.1% |
| `13_apple_healthy.JPG` | Apple — healthy | Shimla, Himachal Pradesh | 2023-06-15 | 95.6% |
| `14_grape_black_rot.JPG` | Grape — Black rot | Nashik, Maharashtra | 2023-11-10 | 91.2% |
| `15_grape_esca_black_measles.JPG` | Grape — Esca (Black Measles) | Nashik, Maharashtra | 2023-11-10 | 92.2% |
| `16_corn_maize_common_rust.JPG` | Corn (maize) — Common rust  | Davangere, Karnataka | 2023-08-10 | 96.2% |
| `18_squash_powdery_mildew.JPG` | Squash — Powdery mildew | Lucknow, Uttar Pradesh | 2023-07-20 | 94.2% |
| `19_peach_bacterial_spot.JPG` | Peach — Bacterial spot | Solan, Himachal Pradesh | 2023-04-15 | 97.2% |
| `20_strawberry_leaf_scorch.JPG` | Strawberry — Leaf scorch | Mahabaleshwar, Maharashtra | 2023-01-15 | 96.6% |
| `21_orange_haunglongbing_citrus_greening.JPG` | Orange — Haunglongbing (Citrus greening) | Nagpur, Maharashtra | 2023-09-10 | 96.5% |
| `22_pepper_bell_bacterial_spot.JPG` | Pepper, bell — Bacterial spot | Kolar, Karnataka | 2023-09-05 | 92.9% |

## Harder classes

These are genuinely confusable and the model is less certain about them.
They are honest test cases, but do not open a presentation on one.

| File | True class | Farm | Date | Model confidence |
|---|---|---|---|---|
| `09_potato_healthy.JPG` | Potato — healthy | Agra, Uttar Pradesh | 2023-11-20 | 60.6% |
| `12_apple_cedar_apple_rust.JPG` | Apple — Cedar apple rust | Shimla, Himachal Pradesh | 2023-06-15 | 55.2% |
| `17_corn_maize_northern_leaf_blight.JPG` | Corn (maize) — Northern Leaf Blight | Davangere, Karnataka | 2023-08-10 | 68.5% |
| `23_cherry_including_sour_powdery_mildew.JPG` | Cherry (including sour) — Powdery mildew | Srinagar, Jammu & Kashmir | 2023-05-20 | 60.5% |