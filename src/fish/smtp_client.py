from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Sequence

from fish.accounts import Account


def send_email(
    account: Account,
    to_addrs: Sequence[str],
    subject: str,
    body: str,
    cc_addrs: Sequence[str] | None = None,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    msg = EmailMessage()
    msg["From"] = account.email
    msg["To"] = ", ".join(to_addrs)
    if cc_addrs:
        msg["Cc"] = ", ".join(cc_addrs)
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg.set_content(body)

    recipients = list(to_addrs) + list(cc_addrs or [])
    with smtplib.SMTP(account.smtp_host, account.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(account.username, account.password)
        smtp.send_message(msg)

    return msg.as_string()
