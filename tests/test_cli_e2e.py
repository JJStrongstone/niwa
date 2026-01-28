"""
End-to-end CLI tests using subprocess and tempfile.

Tests simulate real LLM agent workflows: reading, editing, conflicts,
multi-agent collaboration, error recovery, and edge cases.

Run with: pytest tests/test_cli_e2e.py -v
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def niwa(*args, cwd):
    """Run niwa CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["niwa", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def db(tmp_path):
    """Initialize a niwa database in a temp directory."""
    rc, out, err = niwa("init", ".", cwd=tmp_path)
    assert rc == 0, f"init failed: {err}"
    return tmp_path


# ── Init ────────────────────────────────────────────────────────────────────


class TestInit:
    def test_init_creates_niwa_dir(self, tmp_path):
        rc, out, err = niwa("init", ".", cwd=tmp_path)
        assert rc == 0
        assert (tmp_path / ".niwa").is_dir()
        assert "INITIALIZED" in out

    def test_tree_after_init(self, db):
        rc, out, err = niwa("tree", cwd=db)
        assert rc == 0
        assert "root" in out
        assert "Document" in out

    def test_init_twice_is_safe(self, db):
        """Agent might init a dir that's already initialized."""
        rc, out, err = niwa("init", ".", cwd=db)
        # Should not crash — either succeeds or warns
        assert rc == 0 or "already" in out.lower() or "exists" in out.lower()


# ── Add ─────────────────────────────────────────────────────────────────────


class TestAdd:
    def test_add_node(self, db):
        rc, out, err = niwa("add", "My Section", "--agent", "test_agent", cwd=db)
        assert rc == 0
        assert "NODE_ID:" in out
        assert "h1_0" in out

    def test_add_shows_in_tree(self, db):
        niwa("add", "First", "--agent", "a1", cwd=db)
        rc, out, err = niwa("tree", cwd=db)
        assert rc == 0
        assert "First" in out
        assert "h1_0" in out

    def test_add_with_parent(self, db):
        niwa("add", "Parent", "--agent", "a1", cwd=db)
        rc, out, err = niwa("add", "Child", "--agent", "a1", "--parent", "h1_0", cwd=db)
        assert rc == 0
        assert "NODE_ID:" in out

    def test_add_duplicate_title_warns(self, db):
        niwa("add", "Dupe", "--agent", "a1", cwd=db)
        rc, out, err = niwa("add", "Dupe", "--agent", "a2", cwd=db)
        assert "already exists" in out.lower() or "duplicate" in out.lower() or rc != 0

    def test_add_duplicate_different_parent(self, db):
        """Same title under different parents is allowed."""
        niwa("add", "Parent A", "--agent", "a1", cwd=db)
        niwa("add", "Parent B", "--agent", "a1", cwd=db)
        niwa("add", "Notes", "--agent", "a1", "--parent", "h1_0", cwd=db)
        rc, out, err = niwa("add", "Notes", "--agent", "a1", "--parent", "h1_1", cwd=db)
        assert rc == 0
        assert "NODE_ID:" in out

    def test_add_deep_nesting(self, db):
        """Build a 4-level deep tree like an agent structuring a spec."""
        niwa("add", "Project", "--agent", "a1", cwd=db)
        niwa("add", "Requirements", "--agent", "a1", "--parent", "h1_0", cwd=db)
        niwa("add", "Functional", "--agent", "a1", "--parent", "h2_0", cwd=db)
        rc, out, err = niwa("add", "Auth Flow", "--agent", "a1", "--parent", "h3_0", cwd=db)
        assert rc == 0
        assert "NODE_ID:" in out

        rc, out, err = niwa("tree", cwd=db)
        assert "Project" in out
        assert "Requirements" in out
        assert "Functional" in out
        assert "Auth Flow" in out

    def test_add_many_siblings(self, db):
        """Agent adds many sections at the same level."""
        for i in range(10):
            rc, out, err = niwa("add", f"Section {i}", "--agent", "a1", cwd=db)
            assert rc == 0
            assert "NODE_ID:" in out

        rc, out, err = niwa("tree", cwd=db)
        for i in range(10):
            assert f"Section {i}" in out

    def test_add_with_content_via_file(self, db):
        """Agent writes content to a file then adds via --file."""
        content_file = db / "content.md"
        content_file.write_text("This is the detailed content\nwith multiple lines.")
        rc, out, err = niwa("add", "From File", "--agent", "a1", "--file", str(content_file), cwd=db)
        assert rc == 0

        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert "detailed content" in out

    def test_add_with_stdin(self, db):
        """Agent pipes content via stdin."""
        result = subprocess.run(
            ["niwa", "add", "From Stdin", "--agent", "a1", "--stdin"],
            cwd=db,
            capture_output=True,
            text=True,
            input="Piped content here",
        )
        assert result.returncode == 0

        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert "Piped content" in out

    def test_add_to_nonexistent_parent(self, db):
        """Agent tries to add under a node that doesn't exist."""
        rc, out, err = niwa("add", "Orphan", "--agent", "a1", "--parent", "h1_999", cwd=db)
        assert rc != 0 or "not found" in (out + err).lower()

    def test_add_without_agent(self, db):
        """Agent forgets --agent flag."""
        rc, out, err = niwa("add", "No Agent", cwd=db)
        # Should still work (agent defaults or warns)
        # Just verify it doesn't crash
        assert rc == 0 or "agent" in (out + err).lower()


# ── Read / Edit cycle ──────────────────────────────────────────────────────


class TestReadEditCycle:
    def test_read_then_edit(self, db):
        niwa("add", "Section", "--agent", "a1", cwd=db)

        rc, out, err = niwa("read", "h1_0", "--agent", "a1", cwd=db)
        assert rc == 0
        assert "READ SUCCESSFULLY" in out

        rc, out, err = niwa("edit", "h1_0", "new content", "--agent", "a1", cwd=db)
        assert rc == 0
        assert "EDIT SUCCESSFUL" in out

    def test_edit_updates_version(self, db):
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "v2 content", "--agent", "a1", cwd=db)

        rc, out, err = niwa("tree", cwd=db)
        assert "v2" in out

    def test_peek_no_tracking(self, db):
        niwa("add", "Section", "--agent", "a1", cwd=db)
        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert rc == 0
        assert "Section" in out

    def test_read_nonexistent_node(self, db):
        """Agent tries to read a node that doesn't exist."""
        rc, out, err = niwa("read", "h1_999", "--agent", "a1", cwd=db)
        assert rc != 0 or "not found" in (out + err).lower()

    def test_edit_nonexistent_node(self, db):
        """Agent tries to edit a node that doesn't exist."""
        rc, out, err = niwa("edit", "h1_999", "content", "--agent", "a1", cwd=db)
        assert rc != 0 or "not found" in (out + err).lower()

    def test_multiple_edits_same_agent(self, db):
        """Agent does read-edit-read-edit cycle multiple times."""
        niwa("add", "Iterative", "--agent", "a1", cwd=db)

        for i in range(5):
            rc, out, err = niwa("read", "h1_0", "--agent", "a1", cwd=db)
            assert rc == 0
            rc, out, err = niwa("edit", "h1_0", f"iteration {i}", "--agent", "a1",
                                "--summary", f"edit {i}", cwd=db)
            assert rc == 0
            assert "EDIT SUCCESSFUL" in out

        rc, out, err = niwa("tree", cwd=db)
        assert "v6" in out  # v1 from add + 5 edits

    def test_edit_with_summary(self, db):
        """Agent provides a summary of changes."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        rc, out, err = niwa("edit", "h1_0", "updated", "--agent", "a1",
                            "--summary", "Rewrote introduction", cwd=db)
        assert rc == 0
        assert "EDIT SUCCESSFUL" in out

    def test_edit_content_with_special_chars(self, db):
        """Content with quotes, newlines, markdown formatting."""
        niwa("add", "Special", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        content = 'He said "hello" & used <html> tags'
        rc, out, err = niwa("edit", "h1_0", content, "--agent", "a1", cwd=db)
        assert rc == 0

        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert "hello" in out

    def test_edit_with_file(self, db):
        """Agent writes large content via --file flag."""
        niwa("add", "Large Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        content_file = db / "edit_content.md"
        content_file.write_text("## Subsection\n\nDetailed paragraph.\n\n- Item 1\n- Item 2\n")

        rc, out, err = niwa("edit", "h1_0", "--agent", "a1", "--file", str(content_file), cwd=db)
        assert rc == 0

        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert "Detailed paragraph" in out

    def test_edit_empty_content(self, db):
        """Agent clears a section's content."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "some content", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        rc, out, err = niwa("edit", "h1_0", "", "--agent", "a1", cwd=db)
        assert rc == 0


# ── Conflict detection ──────────────────────────────────────────────────────


class TestConflicts:
    def test_concurrent_edit_conflict(self, db):
        """Two agents read the same version, both try to edit."""
        niwa("add", "Shared", "--agent", "a1", cwd=db)

        # Both agents read v1
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a2", cwd=db)

        # a1 edits first — succeeds
        rc1, out1, _ = niwa("edit", "h1_0", "a1 version", "--agent", "a1", cwd=db)
        assert rc1 == 0

        # a2 tries to edit — should detect conflict
        rc2, out2, err2 = niwa("edit", "h1_0", "a2 version", "--agent", "a2", cwd=db)
        combined = out2 + err2
        assert "conflict" in combined.lower() or "CONFLICT" in combined

    def test_resolve_accept_yours(self, db):
        """Agent resolves conflict by accepting their version."""
        niwa("add", "Shared", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a2", cwd=db)
        niwa("edit", "h1_0", "a1 version", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "a2 version", "--agent", "a2", cwd=db)

        rc, out, err = niwa("resolve", "h1_0", "ACCEPT_YOURS", "--agent", "a2", cwd=db)
        # Should resolve (or indicate no conflict if auto-resolved)
        combined = out + err
        assert rc == 0 or "resolve" in combined.lower()

    def test_resolve_accept_theirs(self, db):
        """Agent resolves conflict by accepting other's version."""
        niwa("add", "Shared", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a2", cwd=db)
        niwa("edit", "h1_0", "a1 version", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "a2 version", "--agent", "a2", cwd=db)

        rc, out, err = niwa("resolve", "h1_0", "ACCEPT_THEIRS", "--agent", "a2", cwd=db)
        combined = out + err
        assert rc == 0 or "resolve" in combined.lower()

    def test_conflicts_command(self, db):
        """List pending conflicts."""
        rc, out, err = niwa("conflicts", cwd=db)
        # Should work even with no conflicts
        assert rc == 0 or "no conflict" in (out + err).lower()

    def test_no_conflict_after_sequential_edit(self, db):
        """Agent reads AFTER previous edit — no conflict."""
        niwa("add", "Section", "--agent", "a1", cwd=db)

        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "first edit", "--agent", "a1", cwd=db)

        # a2 reads the NEW version, then edits — should be clean
        niwa("read", "h1_0", "--agent", "a2", cwd=db)
        rc, out, err = niwa("edit", "h1_0", "second edit", "--agent", "a2", cwd=db)
        assert rc == 0
        assert "EDIT SUCCESSFUL" in out


# ── Export ──────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_markdown(self, db):
        niwa("add", "Title", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "Some content here", "--agent", "a1", cwd=db)

        rc, out, err = niwa("export", cwd=db)
        assert rc == 0
        assert "# Title" in out
        assert "Some content here" in out

    def test_export_preserves_hierarchy(self, db):
        """Export a multi-level tree and verify heading levels."""
        niwa("add", "Top", "--agent", "a1", cwd=db)
        niwa("add", "Mid", "--agent", "a1", "--parent", "h1_0", cwd=db)
        niwa("add", "Bottom", "--agent", "a1", "--parent", "h2_0", cwd=db)

        niwa("read", "h2_0", "--agent", "a1", cwd=db)
        niwa("edit", "h2_0", "mid content", "--agent", "a1", cwd=db)

        rc, out, err = niwa("export", cwd=db)
        assert rc == 0
        assert "# Top" in out
        assert "## Mid" in out
        assert "### Bottom" in out
        assert "mid content" in out

    def test_export_empty_db(self, db):
        """Export right after init — should produce minimal output."""
        rc, out, err = niwa("export", cwd=db)
        assert rc == 0


# ── Search ──────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_finds_node(self, db):
        niwa("add", "Unique Heading", "--agent", "a1", cwd=db)
        rc, out, err = niwa("search", "Unique", cwd=db)
        assert rc == 0
        assert "Unique Heading" in out

    def test_search_no_results(self, db):
        """Search for something that doesn't exist."""
        niwa("add", "Hello", "--agent", "a1", cwd=db)
        rc, out, err = niwa("search", "zzzznonexistent", cwd=db)
        # Should not crash, just show no results
        assert rc == 0 or "no results" in (out + err).lower() or "not found" in (out + err).lower()

    def test_search_finds_content(self, db):
        """Search matches content, not just titles."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "The quantum flux capacitor is broken", "--agent", "a1", cwd=db)

        rc, out, err = niwa("search", "quantum", cwd=db)
        assert rc == 0
        assert "h1_0" in out or "quantum" in out.lower()

    def test_search_case_insensitive_by_default(self, db):
        """Search should be case-insensitive by default."""
        niwa("add", "UPPERCASE TITLE", "--agent", "a1", cwd=db)
        rc, out, err = niwa("search", "uppercase", cwd=db)
        assert rc == 0
        assert "UPPERCASE" in out


# ── Claude Hooks ────────────────────────────────────────────────────────────


class TestClaudeHooks:
    def test_setup_claude_hooks(self, db):
        rc, out, err = niwa("setup", "claude", cwd=db)
        assert rc == 0
        assert "HOOKS INSTALLED" in out

        settings_path = db / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings

    def test_remove_claude_hooks(self, db):
        niwa("setup", "claude", cwd=db)
        rc, out, err = niwa("setup", "claude", "--remove", cwd=db)
        assert rc == 0
        assert "REMOVED" in out

    def test_setup_idempotent(self, db):
        """Running setup twice shouldn't break anything."""
        niwa("setup", "claude", cwd=db)
        rc, out, err = niwa("setup", "claude", cwd=db)
        assert rc == 0
        assert "HOOKS INSTALLED" in out

        settings_path = db / ".claude" / "settings.json"
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings


# ── History ─────────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_shows_edits(self, db):
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "edit 1", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "edit 2", "--agent", "a1", cwd=db)

        rc, out, err = niwa("history", "h1_0", cwd=db)
        assert rc == 0
        assert "Version 1" in out
        assert "Version 2" in out
        assert "Version 3" in out

    def test_history_nonexistent_node(self, db):
        """Agent asks for history of a node that doesn't exist."""
        rc, out, err = niwa("history", "h1_999", cwd=db)
        assert rc != 0 or "not found" in (out + err).lower()

    def test_history_shows_agents(self, db):
        """History should show which agent made each edit."""
        niwa("add", "Shared", "--agent", "alice", cwd=db)
        niwa("read", "h1_0", "--agent", "alice", cwd=db)
        niwa("edit", "h1_0", "alice edit", "--agent", "alice", cwd=db)
        niwa("read", "h1_0", "--agent", "bob", cwd=db)
        niwa("edit", "h1_0", "bob edit", "--agent", "bob", cwd=db)

        rc, out, err = niwa("history", "h1_0", cwd=db)
        assert rc == 0
        assert "alice" in out
        assert "bob" in out


# ── Multi-Agent ─────────────────────────────────────────────────────────────


class TestMultiAgent:
    def test_two_agents_no_conflict(self, db):
        niwa("add", "Sec A", "--agent", "a1", cwd=db)
        niwa("add", "Sec B", "--agent", "a2", cwd=db)

        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("read", "h1_1", "--agent", "a2", cwd=db)

        rc1, out1, _ = niwa("edit", "h1_0", "a1 content", "--agent", "a1", cwd=db)
        rc2, out2, _ = niwa("edit", "h1_1", "a2 content", "--agent", "a2", cwd=db)

        assert rc1 == 0
        assert rc2 == 0
        assert "EDIT SUCCESSFUL" in out1
        assert "EDIT SUCCESSFUL" in out2

    def test_three_agents_building_tree(self, db):
        """Three agents collaboratively build a document structure."""
        niwa("add", "Architecture", "--agent", "lead", cwd=db)
        niwa("add", "Frontend", "--agent", "fe_agent", "--parent", "h1_0", cwd=db)
        niwa("add", "Backend", "--agent", "be_agent", "--parent", "h1_0", cwd=db)
        niwa("add", "Database", "--agent", "be_agent", "--parent", "h1_0", cwd=db)

        # Each agent edits their section
        niwa("read", "h2_0", "--agent", "fe_agent", cwd=db)
        niwa("edit", "h2_0", "React + TypeScript", "--agent", "fe_agent", cwd=db)

        niwa("read", "h2_1", "--agent", "be_agent", cwd=db)
        niwa("edit", "h2_1", "Python + FastAPI", "--agent", "be_agent", cwd=db)

        niwa("read", "h2_2", "--agent", "be_agent", cwd=db)
        niwa("edit", "h2_2", "PostgreSQL", "--agent", "be_agent", cwd=db)

        rc, out, err = niwa("export", cwd=db)
        assert rc == 0
        assert "React" in out
        assert "FastAPI" in out
        assert "PostgreSQL" in out

    def test_agents_list(self, db):
        """List all agents who have interacted with the DB."""
        niwa("add", "A", "--agent", "alice", cwd=db)
        niwa("add", "B", "--agent", "bob", cwd=db)
        niwa("add", "C", "--agent", "charlie", cwd=db)

        rc, out, err = niwa("agents", cwd=db)
        assert rc == 0
        assert "alice" in out
        assert "bob" in out
        assert "charlie" in out


# ── Status ──────────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_after_read(self, db):
        """Agent checks status after reading a node."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        rc, out, err = niwa("status", "--agent", "a1", cwd=db)
        assert rc == 0
        # Should show pending read or agent info
        assert "a1" in out or "h1_0" in out

    def test_status_fresh_agent(self, db):
        """New agent checks status before doing anything."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        rc, out, err = niwa("status", "--agent", "newcomer", cwd=db)
        assert rc == 0

    def test_whoami(self, db):
        """Agent uses whoami to get a suggested name."""
        rc, out, err = niwa("whoami", cwd=db)
        assert rc == 0


# ── Title ───────────────────────────────────────────────────────────────────


class TestTitle:
    def test_rename_title(self, db):
        niwa("add", "Old Name", "--agent", "a1", cwd=db)
        rc, out, err = niwa("title", "h1_0", "New Name", "--agent", "a1", cwd=db)
        assert rc == 0

        rc, out, err = niwa("tree", cwd=db)
        assert "New Name" in out

    def test_rename_nonexistent(self, db):
        rc, out, err = niwa("title", "h1_999", "Name", "--agent", "a1", cwd=db)
        assert rc != 0 or "not found" in (out + err).lower()


# ── Diff ────────────────────────────────────────────────────────────────────


class TestDiff:
    def test_diff_between_versions(self, db):
        niwa("add", "Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "first version content", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)
        niwa("edit", "h1_0", "second version content", "--agent", "a1", cwd=db)

        rc, out, err = niwa("diff", "h1_0", cwd=db)
        # Should show some diff output
        assert rc == 0 or "diff" in (out + err).lower() or "version" in (out + err).lower()


# ── Load ────────────────────────────────────────────────────────────────────


class TestLoad:
    def test_load_markdown_file(self, db):
        """Load a markdown file into the database."""
        md_file = db / "test.md"
        md_file.write_text("# Introduction\n\nHello world.\n\n## Details\n\nSome details.\n")

        rc, out, err = niwa("load", str(md_file), cwd=db)
        assert rc == 0

        rc, out, err = niwa("tree", cwd=db)
        assert "Introduction" in out
        assert "Details" in out

    def test_load_complex_markdown(self, db):
        """Load markdown with multiple heading levels."""
        md_file = db / "complex.md"
        md_file.write_text(
            "# Chapter 1\n\nIntro.\n\n"
            "## Section 1.1\n\nContent.\n\n"
            "## Section 1.2\n\nMore content.\n\n"
            "# Chapter 2\n\nAnother chapter.\n\n"
            "## Section 2.1\n\nDetails.\n"
        )

        rc, out, err = niwa("load", str(md_file), cwd=db)
        assert rc == 0

        rc, out, err = niwa("tree", cwd=db)
        assert "Chapter 1" in out
        assert "Chapter 2" in out
        assert "Section 1.1" in out
        assert "Section 2.1" in out

    def test_load_then_export_roundtrip(self, db):
        """Load markdown, export it, verify structure preserved."""
        md_file = db / "roundtrip.md"
        original = "# Title\n\nParagraph one.\n\n## Subtitle\n\nParagraph two.\n"
        md_file.write_text(original)

        niwa("load", str(md_file), cwd=db)
        rc, out, err = niwa("export", cwd=db)
        assert rc == 0
        assert "# Title" in out
        assert "Paragraph one." in out
        assert "## Subtitle" in out
        assert "Paragraph two." in out


# ── Check ───────────────────────────────────────────────────────────────────


class TestCheck:
    def test_check_healthy_db(self, db):
        """Database health check on a fresh DB."""
        niwa("add", "Section", "--agent", "a1", cwd=db)
        rc, out, err = niwa("check", cwd=db)
        assert rc == 0


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_unknown_command(self, db):
        """Agent types a wrong command — shows guide + error box."""
        rc, out, err = niwa("frobnicate", cwd=db)
        combined = out + err
        # Niwa shows the help guide and an error message for unknown commands
        assert "unknown" in combined.lower() or "error" in combined.lower() or "valid" in combined.lower()

    def test_no_command(self, db):
        """Agent runs niwa with no arguments."""
        rc, out, err = niwa(cwd=db)
        # Should show help/guide, not crash
        assert rc == 0 or "usage" in (out + err).lower() or "niwa" in (out + err).lower()

    def test_help_command(self, db):
        """Agent asks for help."""
        rc, out, err = niwa("help", cwd=db)
        assert rc == 0
        assert "niwa" in out.lower() or "command" in out.lower()

    def test_no_db_initialized(self, tmp_path):
        """Agent tries to use niwa without init."""
        rc, out, err = niwa("tree", cwd=tmp_path)
        assert rc != 0 or "init" in (out + err).lower() or "not found" in (out + err).lower()


# ── Unicode / edge cases ───────────────────────────────────────────────────


class TestUnicodeAndEdgeCases:
    def test_unicode_title(self, db):
        """Japanese, emoji, and special characters in titles."""
        rc, out, err = niwa("add", "設計ドキュメント", "--agent", "a1", cwd=db)
        assert rc == 0

        rc, out, err = niwa("tree", cwd=db)
        assert "設計ドキュメント" in out

    def test_emoji_in_title(self, db):
        rc, out, err = niwa("add", "🚀 Launch Plan", "--agent", "a1", cwd=db)
        assert rc == 0

        rc, out, err = niwa("tree", cwd=db)
        assert "Launch Plan" in out

    def test_very_long_title(self, db):
        long_title = "A" * 500
        rc, out, err = niwa("add", long_title, "--agent", "a1", cwd=db)
        assert rc == 0

    def test_very_long_content(self, db):
        """Agent writes a very large section."""
        niwa("add", "Big Section", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        big_content = "Line of content.\n" * 1000
        content_file = db / "big.md"
        content_file.write_text(big_content)

        rc, out, err = niwa("edit", "h1_0", "--agent", "a1", "--file", str(content_file), cwd=db)
        assert rc == 0

    def test_content_with_markdown_formatting(self, db):
        """Content that itself contains markdown headings."""
        niwa("add", "Guide", "--agent", "a1", cwd=db)
        niwa("read", "h1_0", "--agent", "a1", cwd=db)

        content = "Here is how to use headers:\n\n```markdown\n# This is H1\n## This is H2\n```\n\nDone."
        content_file = db / "guide.md"
        content_file.write_text(content)

        rc, out, err = niwa("edit", "h1_0", "--agent", "a1", "--file", str(content_file), cwd=db)
        assert rc == 0

        rc, out, err = niwa("peek", "h1_0", cwd=db)
        assert "```" in out or "H1" in out
