import os
from typing import Literal, Optional

from openai import OpenAI

MsgType = Literal["recruiter", "senior"]


RECRUITER_TEMPLATE = """Hi {name},
I hope you’re doing well. I’m {me_blurb}. I recently applied for a Software Engineer role at {company} because the work strongly aligns with my experience building full-stack applications and AI-driven solutions.

If possible, I would truly appreciate it if you could take a look at my application. If you think my experience aligns with what you are looking for, I'd love to chat about the position as soon as you are available. If not, please guide me on who I should connect with for this role. I’d be very excited about the opportunity to contribute.

Thank you for your time. I’d be glad to stay connected.

Best regards,
{my_name}
{my_email}
"""

SENIOR_TEMPLATE = """Hi {name},
I hope this isn’t too random. I’m currently exploring opportunities as a software developer at {company}. Since you are part of the team, I was wondering if you could please share some tips on navigating the hiring process or maybe even refer me. Thank you for your time. Your guidance would mean a lot to me.

A bit about me: {me_blurb_long}

Thank you so much for your time. I’d truly value your guidance.

Best regards,
{my_name}
"""


def personalize_message_with_llm(
    msg_type: MsgType,
    *,
    name: str,
    company: str,
    me_blurb: str,
    me_blurb_long: str,
    my_name: str,
    my_email: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Uses LLM ONLY to safely fill the template, keep wording identical, and clean formatting.
    No ranking. No rewriting. Just insert name/company and keep the rest unchanged.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback: purely deterministic formatting if you don't want OpenAI
        return _fill_template(
            msg_type,
            name=name,
            company=company,
            me_blurb=me_blurb,
            me_blurb_long=me_blurb_long,
            my_name=my_name,
            my_email=my_email,
        )

    client = OpenAI(api_key=api_key)

    template = RECRUITER_TEMPLATE if msg_type == "recruiter" else SENIOR_TEMPLATE

    # The instruction is: keep text EXACT except filling placeholders + whitespace normalization
    system = (
        "You are a formatter. Do NOT rewrite any sentences.\n"
        "You must output EXACTLY the template text, with placeholders replaced.\n"
        "Allowed changes: replace placeholders, fix extra spaces, keep line breaks readable.\n"
        "Disallowed: rephrasing, adding/removing sentences, changing tone.\n"
        "Return only the final message, no quotes, no markdown."
    )

    user = {
        "template": template,
        "values": {
            "name": name,
            "company": company,
            "me_blurb": me_blurb,
            "me_blurb_long": me_blurb_long,
            "my_name": my_name,
            "my_email": my_email,
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        temperature=0.0,
    )

    out = (resp.choices[0].message.content or "").strip()
    return out if out else _fill_template(
        msg_type,
        name=name,
        company=company,
        me_blurb=me_blurb,
        me_blurb_long=me_blurb_long,
        my_name=my_name,
        my_email=my_email,
    )


def _fill_template(
    msg_type: MsgType,
    *,
    name: str,
    company: str,
    me_blurb: str,
    me_blurb_long: str,
    my_name: str,
    my_email: str,
) -> str:
    template = RECRUITER_TEMPLATE if msg_type == "recruiter" else SENIOR_TEMPLATE
    return template.format(
        name=name or "there",
        company=company,
        me_blurb=me_blurb,
        me_blurb_long=me_blurb_long,
        my_name=my_name,
        my_email=my_email,
    )