# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-pro-preview-05-06"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""ESTP는 학업에 어떤 자세로 임하는가?"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Here's a thinking process for generating the answer about ESTP's approach to academics:

1.  **Deconstruct the Request:**
    *   The user wants to know how an ESTP personality type approaches academics (학업).
    *   The answer should be based on MBTI knowledge, following the pattern of previous examples (Question about MBTI type -> Concise answer reflecting typical traits).
    *   The previous examples are short and focus on a key characteristic relevant to the situation.

2.  **Analyze the ESTP Type:**
    *   **E (Extraverted):** Energized by the external world, action-oriented, sociable.
    *   **S (Sensing):** Focuses on the concrete, present reality, practical, observant, learns through doing.
    *   **T (Thinking):** Logical, objective, analytical decision-making.
    *   **P (Perceiving):** Flexible, adaptable, spontaneous, prefers options open, dislikes rigid structure.

3.  **Connect ESTP Traits to Academics:**
    *   **Sensing (S):** Prefers hands-on learning, practical application over abstract theory. Might find purely theoretical subjects less engaging. Learns best by *doing*.
    *   **Thinking (T):** Can be logical and analytical when engaged, but the focus is often on practical problem-solving rather than deep theoretical exploration.
    *   **Perceiving (P):** Dislikes strict schedules, deadlines, and rigid study plans. May procrastinate. Prefers flexibility and variety. Can be easily bored by repetitive tasks or long lectures. Might thrive in dynamic learning environments or subjects with immediate relevance.
    *   **Extraverted (E):** Might enjoy group projects or interactive learning but could also be easily distracted by social opportunities.

4.  **Synthesize the Key Academic Approach:**
    *   ESTPs are generally *not* the stereotypical \"bookworm\" or highly theoretical student.
    *   Their strength lies in *practical application* and *hands-on experience*.
    *   They are motivated by *immediate relevance* and *action*.
    *   They can struggle with *long-term planning*, *abstract theory*, and *rigid structures* often found in traditional academic settings.
    *   They might excel in subjects that involve doing, experimenting, or solving real-world problems (e.g., labs, workshops, applied sciences, sports, business simulations).
    *   They might find rote memorization or purely theoretical subjects tedious.
    *   The P preference suggests they might cram or rely on bursts of energy rather than consistent, planned study.

5.  **Formulate a Concise Answer (Korean):**
    *   Initial thoughts:
        *   \"실용적인 것에 집중한다.\" (Focuses on practical things.) - A bit too general.
        *   \"이론보다는 실제 경험을 선호한다.\" (Prefers real experience over theory.) - Good, but maybe slightly long.
        *   \"직접 해보면서 배우는 것을 좋아한다.\" (Likes learning by doing.) - Captures the S and P well.
        *   \"흥미로운 부분에 에너지를 쏟는다.\" (Pours energy into interesting parts.) - Highlights the P and potential inconsistency.
        *   \"벼락치기를 잘한다.\" (Is good at cramming.) - Specific P trait, but maybe not the *core* approach.
        *   \"활동적인 학습 방식을 선호한다.\" (Prefers active learning methods.) - Good summary.

    *   Refining: The core idea is the preference for *action* and *practicality* over passive theory. \"활동적인 학습\" (active learning) or \"직접 부딪혀 배우는 것\" (learning by directly engaging/tackling) seems most fitting. Let's try to combine practicality and action.

    *   Consider the examples: They are short and capture a key tendency.
        *   ESFJ -> 현실적인 해결 방법 (Practical solution)
        *   ISFJ -> 섬세하고 다정하게 케어 (Detailed and warm care)
        *   INTP -> 주변 상황 파악 위해 관찰 (Observes to understand surroundings)
        *   ESFJ -> 계획적으로 움직임 (Moves according to plan)

    *   Applying this style to ESTP & Academics:
        *   \"이론보다는 실제적인 경험을 통해 배우는 것을 선호합니다.\" (Prefers learning through practical experience rather than theory.) This feels accurate and captures the S preference strongly. It also implies the action-oriented nature.

6.  **Final Answer Selection:**
    *   \"이론보다는 실제적인 경험을 통해 배우는 것을 선호합니다.\" (Prefers learning through practical experience rather than theory.) This seems like the best fit, aligning with the ESTP's Sensing preference and action-oriented nature, while matching the concise style of the examples."""),
                types.Part.from_text(text="""답: 이론보다는 실제적인 경험을 통해 배우는 것을 선호합니다."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""- MBTI에 대한 지식을 기반으로, MBTI 질문에 답해보세요.

질문: ESFJ는 문제에 봉착했을때 어떻게 대응하는가?
답: 현실적인 해결 방법을 찾기 위해 노력합니다.

### ###

질문: ISFJ는 연인에게 어떻게 대하는 편인가?
답: 섬세하고 다정하게 케어해주는 편입니다.

### ###

질문: INTP는 사람들이 많은 곳에 가면 어떻게 행동하는가?
답: 주변의 상황을 파악하기 위해 관찰하는 편입니다.

### ###

질문: ESFJ는 충동적인 선택을 많이 하는 편인가?
답: 아니다. 계획적으로 움직이는 편입니다."""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
