# Demo images

Curated leaf photographs for demonstrating the system. Every image is from
the **test split**, so none of them was seen during training.

Each image is paired with a farm that grows its crop and a date inside that
crop's real season **where the weather recorded there favoured that disease**
(a healthy leaf gets a calm week) - the same way the benchmark assigned leaves
to farms. Leaves of one crop are spread across different farms. Regenerate
with `python scripts/pair_test_images.py`.

The dashboard's *Load sample input* and *How it works* use this table.

| File | True class | Farm | Date | Weather for this disease | Model confidence |
|---|---|---|---|---|---|
| `01_tomato_late_blight.JPG` | Late blight (Tomato) | Madanapalle, Andhra Pradesh | 2023-11-10 | 65% | 95.8% |
| `02_tomato_early_blight.JPG` | Early blight (Tomato) | Hooghly, West Bengal | 2023-08-10 | 99% | 92.9% |
| `03_tomato_spider_mites_two-spotted_spider_mite.JPG` | Spider mites (Tomato) | Madanapalle, Andhra Pradesh | 2023-07-15 | 53% | 94.9% |
| `04_tomato_tomato_yellow_leaf_curl_virus.JPG` | Yellow leaf curl virus (Tomato) | Madanapalle, Andhra Pradesh | 2023-07-15 | 55% | 97.9% |
| `05_tomato_septoria_leaf_spot.JPG` | Septoria leaf spot (Tomato) | Nashik, Maharashtra | 2023-07-15 | 99% | 96.4% |
| `06_tomato_healthy.JPG` | Healthy (Tomato) | Kolar, Karnataka | 2023-06-22 | calm week | 96.1% |
| `07_potato_late_blight.JPG` | Late blight (Potato) | Agra, Uttar Pradesh | 2023-12-10 | 82% | 94.6% |
| `08_potato_early_blight.JPG` | Early blight (Potato) | Agra, Uttar Pradesh | 2023-11-10 | 57% | 96.7% |
| `09_potato_healthy.JPG` | Healthy (Potato) | Agra, Uttar Pradesh | 2023-01-15 | calm week | 58.8% |
| `10_apple_apple_scab.JPG` | Apple scab (Apple) | Anantnag, Jammu & Kashmir | 2023-05-30 | 84% | 87.6% |
| `11_apple_black_rot.JPG` | Black rot (Apple) | Manali, Himachal Pradesh | 2023-06-22 | 87% | 89.6% |
| `12_apple_cedar_apple_rust.JPG` | Cedar apple rust (Apple) | Shimla, Himachal Pradesh | 2023-06-22 | 100% | 49.3% |
| `13_apple_healthy.JPG` | Healthy (Apple) | Reckong Peo (Kinnaur), Himachal Pradesh | 2023-05-15 | calm week | 95.9% |
| `14_grape_black_rot.JPG` | Black rot (Grape) | Sangli, Maharashtra | 2023-11-10 | 59% | 91.2% |
| `15_grape_esca_black_measles.JPG` | Esca (black measles) (Grape) | Solapur, Maharashtra | 2023-11-10 | 60% | 93.3% |
| `16_corn_maize_common_rust.JPG` | Common rust (Corn (maize)) | Davangere, Karnataka | 2023-07-15 | 81% | 96.1% |
| `17_corn_maize_northern_leaf_blight.JPG` | Northern Leaf Blight (Corn (maize)) | Chhindwara, Madhya Pradesh | 2023-08-10 | 70% | 71.7% |
| `18_squash_powdery_mildew.JPG` | Powdery mildew (Squash) | Patna, Bihar | 2023-09-10 | 53% | 92.6% |
| `19_peach_bacterial_spot.JPG` | Bacterial spot (Peach) | Solan, Himachal Pradesh | 2023-06-22 | 58% | 95.5% |
| `20_strawberry_leaf_scorch.JPG` | Leaf scorch (Strawberry) | Mahabaleshwar, Maharashtra | 2023-11-10 | 57% | 95.8% |
| `21_orange_haunglongbing_citrus_greening.JPG` | Citrus greening (HLB) (Orange) | Nagpur, Maharashtra | 2023-09-10 | 82% | 96.5% |
| `22_pepper_bell_bacterial_spot.JPG` | Bacterial spot (Pepper, bell) | Agra, Uttar Pradesh | 2023-07-15 | 56% | 92.6% |
| `23_cherry_including_sour_powdery_mildew.JPG` | Powdery mildew (Cherry (including sour)) | Manali, Himachal Pradesh | 2023-06-22 | 56% | 79.8% |
