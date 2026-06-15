from __future__ import annotations

from typing import Any

from cmdline import json_dumps

from fish.accounts import account_by_email, auth_status, load_accounts
from fish.actions import message_archive, message_flag, message_move
from fish.bulk import bulk_action
from fish.classifier import predict_labels, train_classifier
from fish.config import DEFAULT_SYNC_DAYS
from fish.parse import message_row_to_dict
from fish.rank import compute_importance, digest, priority_inbox
from fish.search import search_messages
from fish.smtp_client import send_email
from fish.store import (
    db_conn,
    get_draft,
    get_message_by_id,
    get_thread_messages,
    init_db,
    save_draft,
    sync_status,
)
from fish.sync import sync_account, sync_all
from fish.topics import extract_topics, label_topics_with_llm, list_topics, topic_graph, topic_messages

FISH_INSTRUCTIONS = (
    "Fish provides read/write access to Ken's synced IMAP mail across multiple accounts. "
    "Search before bulk operations. Use fish_bulk_action with dry_run=true first unless "
    "the user explicitly asked to proceed immediately. One message equals one RAG chunk."
)

FISH_TOOL_NAMES = [
    "fish_auth_status",
    "fish_sync_run",
    "fish_sync_status",
    "fish_search",
    "fish_message_get",
    "fish_thread_get",
    "fish_message_move",
    "fish_message_archive",
    "fish_message_flag",
    "fish_bulk_action",
    "fish_compose",
    "fish_send",
    "fish_priority_inbox",
    "fish_digest",
    "fish_topics_extract",
    "fish_topics_list",
    "fish_topic_messages",
    "fish_topic_graph",
    "fish_topics_label",
    "fish_importance_compute",
    "fish_classifier_train",
    "fish_classifier_predict",
    "fish_backfill",
]


def _json(data: Any) -> str:
    return json_dumps(data)


def register_tools(mcp: Any, as_json: bool = True) -> None:
    def out(data: Any) -> Any:
        return _json(data) if as_json else data

    @mcp.tool()
    def fish_auth_status() -> Any:
        """Check account configuration, stored credentials, and OpenAI key."""
        return out(auth_status())

    @mcp.tool()
    def fish_sync_run(days: int = DEFAULT_SYNC_DAYS) -> Any:
        """Sync mail from all configured accounts (default last 90 days)."""
        return out(sync_all(days=days))

    @mcp.tool()
    def fish_sync_status() -> Any:
        """Show per-account folder sync state and message counts."""
        init_db()
        with db_conn() as db:
            return out(sync_status(db))

    @mcp.tool()
    def fish_search(
        query: str,
        account_email: str | None = None,
        folder: str | None = None,
        unread_only: bool = False,
        limit: int = 20,
    ) -> Any:
        """Hybrid semantic + keyword search over synced messages."""
        return out(search_messages(query, account_email, folder, unread_only, limit))

    @mcp.tool()
    def fish_message_get(message_id: int) -> Any:
        """Get a full message by local database id."""
        init_db()
        with db_conn() as db:
            row = get_message_by_id(db, message_id)
            if not row:
                return out({"error": f"Message {message_id} not found"})
            acct = db.execute(
                "SELECT email FROM accounts WHERE id = ?", (row["account_id"],)
            ).fetchone()
            item = message_row_to_dict(row)
            item["account_email"] = acct["email"] if acct else None
            return out(item)

    @mcp.tool()
    def fish_thread_get(message_id: int) -> Any:
        """Get all messages in the same thread as the given message."""
        init_db()
        with db_conn() as db:
            return out(get_thread_messages(db, message_id))

    @mcp.tool()
    def fish_message_move(message_id: int, target_folder: str) -> Any:
        """Move a single message to another IMAP folder."""
        return out(message_move(message_id, target_folder))

    @mcp.tool()
    def fish_message_archive(message_id: int) -> Any:
        """Archive a single message using the account's archive folder."""
        return out(message_archive(message_id))

    @mcp.tool()
    def fish_message_flag(message_id: int, flag: str = "\\Flagged", add: bool = True) -> Any:
        """Add or remove an IMAP flag on a message."""
        return out(message_flag(message_id, flag, add))

    @mcp.tool()
    def fish_bulk_action(
        query: str,
        action: str,
        target_folder: str | None = None,
        dry_run: bool = True,
        limit: int = 100,
        account_email: str | None = None,
    ) -> Any:
        """Search then archive/move/flag/delete matching messages. Use dry_run first."""
        return out(bulk_action(query, action, target_folder, dry_run, limit, account_email))

    @mcp.tool()
    def fish_compose(
        account_email: str,
        to_addrs_json: str,
        subject: str,
        body: str,
        cc_addrs_json: str = "[]",
        in_reply_to: str | None = None,
    ) -> Any:
        """Create a local draft. to_addrs_json and cc_addrs_json are JSON string arrays."""
        init_db()
        to_addrs = json.loads(to_addrs_json)
        cc_addrs = json.loads(cc_addrs_json)
        with db_conn() as db:
            draft_id = save_draft(
                db, account_email, to_addrs, subject, body, cc_addrs, in_reply_to
            )
            draft = get_draft(db, draft_id)
        return out({"draft_id": draft_id, "draft": draft})

    @mcp.tool()
    def fish_send(
        account_email: str,
        to_addrs_json: str,
        subject: str,
        body: str,
        cc_addrs_json: str = "[]",
        in_reply_to: str | None = None,
        references: str | None = None,
    ) -> Any:
        """Send an email via SMTP from the given account."""
        account = account_by_email(account_email)
        if not account:
            return out({"error": f"Unknown account {account_email}"})
        to_addrs = json.loads(to_addrs_json)
        cc_addrs = json.loads(cc_addrs_json)
        send_email(account, to_addrs, subject, body, cc_addrs, in_reply_to, references)
        return out({"ok": True, "account_email": account_email, "to": to_addrs, "subject": subject})

    @mcp.tool()
    def fish_priority_inbox(limit: int = 20) -> Any:
        """Return messages ranked by computed importance signals."""
        return out(priority_inbox(limit))

    @mcp.tool()
    def fish_digest(limit: int = 10) -> Any:
        """Summarize top-priority messages and frequent senders."""
        return out(digest(limit))

    @mcp.tool()
    def fish_topics_extract(k: int = 8, limit: int = 200) -> Any:
        """Cluster recent messages into topics."""
        return out(extract_topics(k=k, limit=limit))

    @mcp.tool()
    def fish_topics_list() -> Any:
        """List extracted email topics."""
        return out(list_topics())

    @mcp.tool()
    def fish_topic_messages(topic_id: int, limit: int = 50) -> Any:
        """List messages belonging to a topic."""
        return out(topic_messages(topic_id, limit))

    @mcp.tool()
    def fish_topic_graph() -> Any:
        """Export topic/person graph JSON for mind-map style exploration."""
        return out(topic_graph())

    @mcp.tool()
    def fish_topics_label() -> Any:
        """Use LLM to improve topic cluster labels."""
        return out(label_topics_with_llm())

    @mcp.tool()
    def fish_importance_compute(limit: int = 500) -> Any:
        """Recompute importance scores for recent messages."""
        return out({"updated": compute_importance(limit)})

    @mcp.tool()
    def fish_classifier_train(labels_json: str, name: str = "folder_classifier") -> Any:
        """Train a folder classifier. labels_json maps message_id strings to folder labels."""
        labels = {int(k): v for k, v in json.loads(labels_json).items()}
        return out(train_classifier(labels, name))

    @mcp.tool()
    def fish_classifier_predict(message_ids_json: str, name: str = "folder_classifier") -> Any:
        """Predict folder labels for message ids."""
        ids = [int(x) for x in json.loads(message_ids_json)]
        return out(predict_labels(ids, name))

    @mcp.tool()
    def fish_backfill(since_date: str, account_email: str | None = None) -> Any:
        """Backfill mail older than the default sync window. since_date: YYYY-MM-DD."""
        since = date.fromisoformat(since_date)
        accounts = load_accounts()
        if account_email:
            accounts = [a for a in accounts if a.email == account_email]
        results = [sync_account(a, since=since) for a in accounts]
        return out(results)
