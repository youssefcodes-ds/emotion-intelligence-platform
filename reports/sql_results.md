# SQL query results

## q01_class_distribution_by_split

| split      | label_name   | valence   |   n_messages |   pct_of_split |
|:-----------|:-------------|:----------|-------------:|---------------:|
| test       | joy          | positive  |          695 |          34.75 |
| test       | sadness      | negative  |          581 |          29.05 |
| test       | anger        | negative  |          275 |          13.75 |
| test       | fear         | negative  |          224 |          11.2  |
| test       | love         | positive  |          159 |           7.95 |
| test       | surprise     | ambiguous |           66 |           3.3  |
| train      | joy          | positive  |         5356 |          33.51 |
| train      | sadness      | negative  |         4665 |          29.19 |
| train      | anger        | negative  |         2159 |          13.51 |
| train      | fear         | negative  |         1934 |          12.1  |
| train      | love         | positive  |         1298 |           8.12 |
| train      | surprise     | ambiguous |          571 |           3.57 |
| validation | joy          | positive  |          704 |          35.2  |
| validation | sadness      | negative  |          550 |          27.5  |
| validation | anger        | negative  |          275 |          13.75 |

## q02_corpus_summary

|   total_messages |   n_splits |   n_classes |   avg_words |   min_words |   max_words |   avg_chars |   pct_with_negation |
|-----------------:|-----------:|------------:|------------:|------------:|------------:|------------:|--------------------:|
|            19983 |          3 |           6 |       19.14 |           2 |          66 |       96.69 |               22.74 |

## q03_length_percentiles_by_emotion

| label_name   |   n_messages |   p50_words |   p90_words |   max_words |
|:-------------|-------------:|------------:|------------:|------------:|
| love         |         1635 |          19 |          35 |          63 |
| anger        |         2709 |          17 |          34 |          62 |
| joy          |         6755 |          17 |          34 |          64 |
| surprise     |          718 |          17 |          34 |          61 |
| fear         |         2370 |          16 |          33 |          60 |
| sadness      |         5796 |          16 |          33 |          66 |

## q04_negation_rate_by_emotion

| label_name   | valence   |   n_messages |   n_with_negation |   pct_with_negation |
|:-------------|:----------|-------------:|------------------:|--------------------:|
| anger        | negative  |         2709 |               699 |               25.8  |
| sadness      | negative  |         5796 |              1401 |               24.17 |
| love         | positive  |         1635 |               370 |               22.63 |
| fear         | negative  |         2370 |               504 |               21.27 |
| joy          | positive  |         6755 |              1423 |               21.07 |
| surprise     | ambiguous |          718 |               148 |               20.61 |

## q05_valence_breakdown

| split      | valence   |   n_messages |   avg_words |
|:-----------|:----------|-------------:|------------:|
| test       | negative  |         1080 |        18.9 |
| test       | positive  |          854 |        19.4 |
| test       | ambiguous |           66 |        19.8 |
| train      | negative  |         8758 |        18.7 |
| train      | positive  |         6654 |        19.7 |
| train      | ambiguous |          571 |        20   |
| validation | negative  |         1037 |        18.5 |
| validation | positive  |          882 |        19.4 |
| validation | ambiguous |           81 |        17.9 |

## q06_longest_message_per_emotion

| label_name   |   rank_in_class |   n_words | text                                                                                                                                                                                                                                                                                                        |
|:-------------|----------------:|----------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| anger        |               1 |        62 | i feel not offended in any form and should not make this big and in the end it doesnt bother me at all but ive learned to show some balls in the past and say what i think not anonymous so if we would give some weight to the content of these comments there would be the questions what is behind it    |
| anger        |               2 |        59 | i needed a plan on how to get rid of that feeling it was totally taking over everything i am totally distracted at work with everything i m trying to do in any free time i have in the evenings the projects are taking over my life and the fact that i totally feel burnt out by it all                  |
| anger        |               3 |        59 | i want to tell him how i feel how disgusted i am that he can hurt my husband the way he does and then just laugh about it how he treats his grandchildren how he treated my husbands mum and just scream at him to stop being such a selfish bastard because the world does not revolve around him          |
| fear         |               1 |        60 | i was not going to be able to sleep until i knew how it ended and mostly because of another thing which i am not even going to talk about here because it makes me angry all over again and also because i feel horribly neurotic and immature getting upset about it and so we will gloss over that bit    |
| fear         |               2 |        59 | i know who all think this way so i ve always feel skeptical about painting my nails red since i also have light skin so the red is really going to stand out is there a cute way for a year old to wear red nails without looking like she s trying too hard or looking like a hooker                       |
| fear         |               3 |        59 | i was to worried about them knowing if i was high or not and feeling a little paranoid and i have never never been that type of person that would think and care about what people think about me and would always focus on what i had to do to get to where i needed to get in life                        |
| joy          |               1 |        64 | i lost my special mind but don t worry i m still sane i just wanted you to feel what i felt while reading this book i don t know how many times it was said that sam was special but i can guarantee you it was many more times than what i used in that paragraph did i tell you she was special           |
| joy          |               2 |        64 | i am happier this year in all ways i am just glad i am on english lit only i made good module choices i like my teachers the peeps in my class are not so snidey i feel more confident in my work and i am on top of it unlike last year when i was soooooooooooo behind to the point of doing zero         |
| joy          |               3 |        64 | i feel you i dont believ in you but i keep my faithful to you god gives me a chance to feel what is apathetic after it but much apathetic open up my mind that i can hide this feeling for you i know youre playing with me you show off your love like and maybe after it youll be gone will it happens    |
| love         |               1 |        63 | i confused my feelings with the truth because i liked the view when there was me and you i cant believe that i could be so blind its like you were floating when i was falling and i didnt mind because i like the view i thought you felt it too when there was me and you lyrics from a href http www     |
| love         |               2 |        62 | i like to add a slice of cheese and some pepper to the egg and when i am feeling naughty i like to add some chocolate chips to my trail mix another treat i am loving as a pregnant mom who often craves a sweet but doesn t want to overload on sugar or empty calories is zico coconut water in chocolate |
| love         |               3 |        58 | i do not agree with hirsi ali on policy matters and i do agree with much of what ingrid writes by contrast but having grown up in a country for which i feel little love and with the culture of which i do not identify in the least i can t help but to be sympathetic to her                             |
| sadness      |               1 |        66 | i guess which meant or so i assume no photos no words or no other way to convey what it really feels unless you feels it yourself or khi bi t au th m i bi t th ng ng i b au i rephrase it to a bit more gloomy context unless you are hurt yourself you will never have sympathy for the hurt ones         |
| sadness      |               2 |        64 | i feel in my bones like nobody cares if im here nobody cares if im gone here i am again saying im feeling so lonely people either say its ok to be alone or just go home it kills me and i dont know why it doesnt mean i dont try i try and try but people just treat me like im a ghost                   |
| sadness      |               3 |        61 | i feel rotten all week because i hardly ever see you that s why i wrote this hopeless song i ve never been in love with a girl like you before darling come with me such a wonderful thing has never happened to me before you re the only one who touched my heart it s all a question of courage          |

## q07_imbalance_ratio

| label_name   |   n_train |   size_rank |   pct_of_train |   times_smaller_than_majority |
|:-------------|----------:|------------:|---------------:|------------------------------:|
| joy          |      5356 |           1 |          33.51 |                          1    |
| sadness      |      4665 |           2 |          29.19 |                          1.15 |
| anger        |      2159 |           3 |          13.51 |                          2.48 |
| fear         |      1934 |           4 |          12.1  |                          2.77 |
| love         |      1298 |           5 |           8.12 |                          4.13 |
| surprise     |       571 |           6 |           3.57 |                          9.38 |

## q08_cumulative_class_share

| label_name   |    n |   cumulative_pct |
|:-------------|-----:|-----------------:|
| joy          | 6755 |            33.8  |
| sadness      | 5796 |            62.81 |
| anger        | 2709 |            76.36 |
| fear         | 2370 |            88.22 |
| love         | 1635 |            96.41 |
| surprise     |  718 |           100    |

## q09_model_accuracy_comparison

_No rows - the source table is not populated yet._

## q10_per_class_recall_by_model

_No rows - the source table is not populated yet._

## q11_most_confused_pairs

_No rows - the source table is not populated yet._

## q12_low_confidence_review_queue

_No rows - the source table is not populated yet._

## q13_emotion_topic_matrix

_No rows - the source table is not populated yet._

## q14_inference_latency_by_model

_No rows - the source table is not populated yet._
