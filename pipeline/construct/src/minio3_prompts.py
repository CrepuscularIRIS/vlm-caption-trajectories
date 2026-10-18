"""Mini-o3 cold-start generation prompts, transcribed from arXiv 2509.07969 App. E-F.

Tables 5-8 are verbatim. The four in-context exemplars are the thought/action text of
Figs. 9-12; their images are not published, so each image slot is an explicit placeholder
(recorded deviation). Exemplar 4 is re-ordered into turn order (the PDF layout scrambles it)
and its final thought, which is not printed in Fig. 12, is omitted.
"""

TABLE6_SYSTEM = """You are an agent that answers the questions by using following two actions:
1. grounding([x1, y1, x2, y2], source): Return the bounding box [w1, h1, w2, h2] for the region on the 'source' observation image, where (w1, h1) and (w2, h2) are the top-left and bottom-right coordinates (note: w and h represent the width and height; the width and height of the image are normalized to 1.0; the coordinates of top-left corner are (0, 0).), and 'source' could be either "original image" or "observation i", meaning one of the history observations;
2. final(answer): return the final answer.
Your generation should have the following format:
Thought: to analyze the question & observation and inform the action.
Action: your action
Here are several examples:"""

TABLE8_INSTRUCTION = """Please follow the language style of the above examples to finish the thought and action.
You should mimic human to think carefully and comprehensively. Output the thought as detailed as possible, along with some useful description of the observation, suspective conjectures, and underlying world knowledge that might be useful for problem solving.
If the target object is present in observation i, please analyze whether final answer is ready or further grounding is needed.
Otherwise, you can go back to either original image or any of the history observations to find another region. In this case, clarify this thinking process, and analyze how the old bounding box deviates from the correct one, and how you obtain the coordinates of the new bounding box in Thought i.
The output format should be in JSON as:
{
"thought": "xxx",
"action": "grounding([x1, y1, x2, y2], source='xxx')"
}"""

TABLE5_JUDGE_SYSTEM = """You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
INSTRUCTIONS:
- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer."""

TABLE5_JUDGE_USER = """I will give you a question related to an image and the following text as inputs:
1. **Question Related to the Image**: {question}
2. **Ground Truth Answer**: {gold}
3. **Model Predicted Answer**: {pred}
Your task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:
- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?
- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:
(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.
(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.
**Output Format**:
Your response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.
The format should be Score: 0 or 1"""

IMG = "[image not shown in this example]"

# (question, [(thought, action), ...]); observation i follows action i-1.
EXEMPLARS = [
    ("What is written in English on the top first line of the right side of the wooden wall in the middle of the picture?", [
        ("The user's question asks about the text on the right side of the wooden wall. To analyze it, I may need to zoom into the center part of the image. It seems like the text is too small, so I'll have to crop that region and possibly enlarge the image to read it properly. It could be something like \"meeting room\" or \"conference room,\" but I'll need to visually inspect and confirm. I see there might be small metal letters on the wooden wall, likely on the right side. To check the text, I'll crop the image around the dimensions of 0.45-0.65 x 0.3-0.6. After cropping, I'll zoom in to make it easier to read.",
         'grounding([0.45,0.3,0.65,0.6], source="original_image")'),
        ("I see small metal letters on the right side of the central wooden wall. To focus on just the text, I'll crop the image more narrowly, focusing only on the text area. By narrowing the crop horizontally, I can zoom in specifically on the text, which should help me read it more clearly. The target text is located approximately at the top half of this image, so let's focus on the range from 0.25 to 0.5 of the height dimension in the image of observation 1.",
         'grounding([0.05,0.25,0.95,0.50], source="observation_1")'),
        ("I noticed there might be small metal letters on the right side of the central wooden wall, which is facing the camera. But I could only see the first line of English text partially in observation 2. So I should return to observation 1 and try a better bounding box. The previous bounding box is [0.05,0.25,0.95,0.50], and the resulting observation seems too high, so I should move the top left corner of the bounding box downwards from (0.05, 0.25) to (0.05, 0.35), and the bottom right corner from (0.95, 0.50) to (0.95, 0.55). Therefore, the new bounding box should be [0.05,0.35,0.95,0.55] to better see the text.",
         'grounding([0.05,0.35,0.95,0.55], source="observation_1")'),
        ("Upon closer inspection, I can clearly see the English text on the top first line is \"Meeting Room\". The final answer is confirmed. No further checking is needed.",
         'final("Meeting Room")'),
    ]),
    ("What are the black numbers on the yellow license plate?", [
        ("The user is asking about black numbers on a yellow license plate in an image. Let me locate that plate. It looks like it's a small yellow rectangle on a scooter or cart, on the center-left of the image behind a walking man. The plate seems to be in the central region, around the left-center, behind a man on a scooter or tricycle. I'll try cropping that central area for a closer look. I'll crop from coordinates around 0.4-0.6 horizontally and 0.4-0.7 vertically.",
         'grounding([0.4,0.4,0.6,0.7], source="original_image")'),
        ("It looks like the plate is behind a man on a scooter or tricycle, and it's toward the left-center, identifying it as a Thai plate with blue and yellow colors. I'll adjust the cropping slightly lower to refine the area. The license plate is approximately at the bottom-left region, around 0.0-0.4 horizontally and 0.6-0.8 vertically. I'll zoom in on this area.",
         'grounding([0.0,0.6,0.4,0.8], source="observation_1")'),
        ("The user wants to know what the black numbers on the yellow license plate are. I can see a \"6\" clearly. It might be a part of a longer number, but the surrounding characters, possibly in Thai, are obscuring the rest. I'll crop further right to check if other digits are visible past the \"6\" that's currently uncovered. The previous bounding box is [0.0,0.6,0.4,0.8], and we need to extend the bounding box further right. The bottom-right corner of the bounding box is (0.4,0.8), so we should set the width from 0.4 to a larger number, possibly 0.8. Accordingly, the new bounding box should be [0.0,0.6,0.8,0.8].",
         'grounding([0.0,0.6,0.8,0.8], source="observation_1")'),
        ("I can see that the only visible number is \"6.\" There seem to be other Thai characters beside it, and possibly more digits to the right, but they're concealed by the man in the image. I double-checked, and I don't see alternate digits or numbers. The only black digit that's visible on that yellow license plate is a \"6.\" Any other numbers that might be on the plate are blocked by the person standing in front of it.",
         'final("6")'),
    ]),
    ("How many people are there in total under the two yellow facilities beneath the building on the right?", [
        ("This is an object recognition and counting problem. I need to count the number of people under the two yellow facilities located beneath the building on the right side of the image. First, I located the building on the right side of the image. It's a building that looks like a church, with a tall tower. Beneath this building, I found two yellow facilities, which look like temporary enclosures or a construction area. I guess the region (0.6-1.0 in width and 0.2-0.8 in height) should include the two yellow facilities. Let's zoom in to see more clearly.",
         'grounding([0.6,0.2,1.0,0.8], source="original_image")'),
        ("I noticed two yellow facilities beneath the church-like building. Several people are seemingly sitting there and they might be chatting. There might be three or four people in total, but I couldn't be certain as the region is blurry in this image. To count more accurately, I need to zoom in on the area where these two yellow facilities are located, based on observation 1. They appear in the central region of observation 1, approximately 0.3-0.6 horizontally and 0.5-0.8 vertically.",
         'grounding([0.3,0.5,0.6,0.8], source="observation_1")'),
        ("In this further zoomed-in image, I could see the two yellow facilities more clearly. Under each facility, there might be two people talking. Maybe four people in total are seemingly over there. But I couldn't be certain as this region looks blurry up to now. I may need a closer look based on observation 2 to verify. The target area is located at the central of this image, so I'll crop this area (0.4-0.8 horizontally, 0.4-0.8 vertically) and see more details.",
         'grounding([0.4,0.4,0.8,0.8], source="observation_2")'),
        ("It is clear now. Two people are talking under one yellow facility and another two people are under the other. This verifies that my previous judgement is correct. Now, I can confirm that there are a total of 4 people under the yellow facilities.",
         'final("4")'),
    ]),
    ("What is the number below the string 207784?", [
        ("Identify that this is a number recognition task requiring determining the digits in the line below \"207784.\" Based on my experience, \"207784\" may be a license plate number. I observe an abandoned truck with its front facing the camera, located in the central part of the image. I'll crop the front part of this truck to check whether the above hypothesis is correct or not. The front part of the truck is in the center of the image, approximately 0.3-0.8 horizontally and 0.45-0.95 vertically.",
         'grounding([0.3,0.45,0.8,0.95], source="original_image")'),
        ("The front part of the truck can be seen in this image. However, there is no license plate number here. Did the bounding box focus on an incorrect region? Wait, let me analyze. I notice the truck lights are present in this crop, including both the left one and the right one. I can confirm that this is indeed the front of the truck. Up to now, we confirm that there are no digits \"207784\" around the abandoned truck. Next, I should continue to search for other areas to locate the target string. I notice there are numbers in the top-right part of the original image, approximately 0.55-0.85 in width and 0-0.25 in height. Let's zoom in on that area based on the original image.",
         'grounding([0.55,0.0,0.85,0.25], source="original_image")'),
        ("I notice two lines of white characters in this crop, and they are \"EMCU3648519\" and \"22G1\". The user asks for the number below \"207784\". Neither matches the target digit string. So I need to continue searching for the target number. I observe many white characters in the top-left corner of the original image, which might include the number being sought. Now, let's zoom in on the top-left part (0.0-0.25 horizontally, 0.0-0.5 vertically) of the original image.",
         'grounding([0.0,0.0,0.25,0.5], source="original_image")'),
        ("In this crop, I observe many white numbers on the left side, which might include the digits being sought. My target is to find the digits that match \"207784\". Let's zoom in on the top left corner of observation 3. Based on observation 3, the region is approximately 0.0-0.3 in width and 0.1-0.4 in height.",
         'grounding([0.0,0.1,0.3,0.4], source="observation_3")'),
        ("I notice that there are two lines of numbers in the bottom left. They look a little bit blurry, probably \"85093922\" and \"22G1\". I am not certain, but one thing to be sure is that neither of them matches the target number \"207784\". I notice there is a number \"4\" at the top of observation 4, which could be part of the number \"207784\". I need to look a little upwards to see the entire digit sequence. Let's return to observation 3, and move the previous bounding box a little upwards. The previous bounding box was [0.0,0.1,0.3,0.4]. What about moving it upwards by 10% to [0.0,0.0,0.3,0.3]?",
         'grounding([0.0,0.0,0.3,0.3], source="observation_3")'),
        ("In this crop, I see a vertical line \"SU 207784 6\" in the top part. The characters seem a little blurry. The number below looks like \"6\". I need to double-check by further zooming in on this area. It looks like in the upper part of observation 5, approximately 0.3-0.7 horizontally and 0.0-0.6 vertically.",
         'grounding([0.3,0.0,0.7,0.6], source="observation_5")'),
        ("A vertical line reads \"SU 207784 6\" in this crop, which aligns with my previous judgement, even though it's a little blurry. I can confirm that the number sequence is \"207784\", and the number directly below it is \"6\".",
         'final("6")'),
    ]),
]


def exemplar_block() -> str:
    """Render the in-context examples in the Table 7 format."""
    parts = []
    for n, (question, steps) in enumerate(EXEMPLARS, 1):
        lines = [f"Example {n}:", f"Question: {question}", f"Original Image: {IMG}"]
        for i, (thought, action) in enumerate(steps):
            lines += [f"Thought {i}: {thought}", f"Action {i}: {action}"]
            if action.startswith("grounding"):
                lines.append(f"Observation {i + 1}: {IMG}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def system_prompt() -> str:
    return TABLE6_SYSTEM + "\n\n" + exemplar_block()


def turn_instruction(i: int) -> str:
    return f"Thought {i}:\nAction {i}:\n{TABLE8_INSTRUCTION}"
