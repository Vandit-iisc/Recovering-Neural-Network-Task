# The GPT-6 Astra single-call baseline

`prompt.txt` is the exact prompt: the network conventions, the question, one labelled example (sample index 43, a handwritten 0, class 1), and every weight in full precision.

## Settings

- Model: GPT-6 Astra via the OpenAI Responses API, high reasoning effort.
- `tools = []`, `tool_choice = none`, `truncation = disabled`.
- **Request 1:** 4,096-token output budget. Used it all on reasoning and ended `incomplete` (`max_output_tokens`) with no answer.
- **Request 2:** fresh request (no prior response attached), 32,768-token budget, visible answer limited to under 300 words. Used 669,642 input and 14,546 output tokens (14,224 reasoning) and completed.

## Result

Class 1 = {0, 2, 4, 6, 8}, class 0 = {1, 3, 5, 7, 9} (parity), confidence 0.42. That puts 4 of 10 digits on the correct side.

## Responses

Put both raw API responses in `responses/`, for example `request1_incomplete.json` and `request2_complete.json`.
