# -*- coding: utf-8 -*-
"""Unit tests for the pure/module-level helper functions in main.py."""
import tarfile
import zipfile

import pytest


# --- id / filename helpers ---------------------------------------------------

def test_sanitize_id_replaces_spaces_and_special_chars(app):
  assert app.sanitize_id("My App! v2.0") == "My_App__v2.0"


def test_sanitize_id_empty_falls_back_to_app(app):
  assert app.sanitize_id("   ") == "app"
  assert app.sanitize_id("") == "app"


@pytest.mark.parametrize(
    "filename,expected_stem",
    [
      ("myapp.tar.gz", "myapp"),
      ("myapp.tgz", "myapp"),
      ("myapp.tar", "myapp"),
      ("myapp.zip", "myapp"),
      ("myapp.7z", "myapp"),
      ("my.app.zip", "my.app"),
    ],
)
def test_strip_archive_ext(app, filename, expected_stem):
  assert app.strip_archive_ext(filename) == expected_stem


@pytest.mark.parametrize(
    "filename,expected_ext",
    [
      ("myapp.tar.gz", ".tar.gz"),
      ("myapp.tgz", ".tgz"),
      ("myapp.zip", ".zip"),
      ("myapp.7z", ".7z"),
    ],
)
def test_archive_ext_of_preserves_multi_part_extension(app, filename, expected_ext):
  # Regression test: Path(filename).suffix would truncate ".tar.gz" to ".gz",
  # which broke extraction of the downloaded temp file.
  assert app.archive_ext_of(filename) == expected_ext


# --- entry point detection ----------------------------------------------------

def test_pick_entry_prefers_main_py(app):
  entry, ambiguous = app.pick_entry_from_candidates(["utils.py", "main.py", "sub/main.py"])
  assert entry == "main.py"
  assert ambiguous is False


def test_pick_entry_single_file_is_unambiguous(app):
  entry, ambiguous = app.pick_entry_from_candidates(["bot.py"])
  assert entry == "bot.py"
  assert ambiguous is False


def test_pick_entry_multiple_files_no_main_is_ambiguous(app):
  entry, ambiguous = app.pick_entry_from_candidates(["a.py", "b.py"])
  assert entry is None
  assert ambiguous is True


def test_pick_entry_no_files(app):
  entry, ambiguous = app.pick_entry_from_candidates([])
  assert entry is None
  assert ambiguous is False


def test_pick_entry_matches_name_hint_when_no_main_py(app):
  entry, ambiguous = app.pick_entry_from_candidates(
      ["utils.py", "myrepo.py"], name_hints=["myrepo"]
  )
  assert entry == "myrepo.py"
  assert ambiguous is False


def test_pick_entry_name_hint_is_case_insensitive(app):
  entry, ambiguous = app.pick_entry_from_candidates(
      ["utils.py", "MyRepo.py"], name_hints=["myrepo"]
  )
  assert entry == "MyRepo.py"
  assert ambiguous is False


def test_pick_entry_main_py_still_wins_over_name_hint(app):
  entry, ambiguous = app.pick_entry_from_candidates(
      ["myrepo.py", "main.py"], name_hints=["myrepo"]
  )
  assert entry == "main.py"
  assert ambiguous is False


def test_pick_entry_still_ambiguous_when_hint_does_not_match(app):
  entry, ambiguous = app.pick_entry_from_candidates(
      ["a.py", "b.py"], name_hints=["myrepo"]
  )
  assert entry is None
  assert ambiguous is True


# --- GitHub URL parsing -------------------------------------------------------

def test_parse_github_url_plain_repo(app):
  info = app.parse_github_url("https://github.com/mrmigles/myapp")
  assert info == {
      "owner": "mrmigles",
      "repo": "myapp",
      "branch": None,
      "path": None,
      "app_id": "myapp",
  }


def test_parse_github_url_with_subfolder(app):
  info = app.parse_github_url("https://github.com/mrmigles/myrepo/tree/main/myapp")
  assert info["owner"] == "mrmigles"
  assert info["repo"] == "myrepo"
  assert info["branch"] == "main"
  assert info["path"] == "myapp"
  # Exact example from the spec: name becomes "myrepo-myapp"
  assert info["app_id"] == "myrepo-myapp"


def test_parse_github_url_with_nested_subfolder_uses_last_segment(app):
  info = app.parse_github_url("https://github.com/owner/myrepo/tree/dev/sub/folder")
  assert info["path"] == "sub/folder"
  assert info["app_id"] == "myrepo-folder"


def test_parse_github_url_rejects_non_github(app):
  assert app.parse_github_url("https://example.com/owner/repo") is None
  assert app.parse_github_url("not a url") is None
  assert app.parse_github_url("") is None


# --- project tree scanning ----------------------------------------------------

def _make_project(tmp_path):
  tmp_path.mkdir(parents=True, exist_ok=True)
  (tmp_path / "sub").mkdir()
  (tmp_path / "main.py").write_text("import os\nos.getenv('FOO')\nos.environ['BAR']\n")
  (tmp_path / "sub" / "helper.py").write_text("x = 1\n")
  (tmp_path / "requirements.txt").write_text("requests\n# a comment\nflask==2.0\n\n")
  ignored = tmp_path / "__pycache__"
  ignored.mkdir()
  (ignored / "ignored.py").write_text("should not be discovered\n")
  return tmp_path


def test_discover_python_files_skips_ignored_dirs(app, tmp_path):
  project = _make_project(tmp_path)
  files = app.discover_python_files(project)
  assert files == ["main.py", "sub/helper.py"]


def test_find_requirements_file_prefers_root(app, tmp_path):
  project = _make_project(tmp_path)
  found = app.find_requirements_file(project)
  assert found == project / "requirements.txt"


def test_find_requirements_file_falls_back_to_nested(app, tmp_path):
  (tmp_path / "sub").mkdir()
  (tmp_path / "sub" / "requirements.txt").write_text("requests\n")
  found = app.find_requirements_file(tmp_path)
  assert found == tmp_path / "sub" / "requirements.txt"


def test_find_requirements_file_none_when_missing(app, tmp_path):
  assert app.find_requirements_file(tmp_path) is None


def test_find_requirements_pkgs_strips_comments_and_blanks(app, tmp_path):
  project = _make_project(tmp_path)
  assert app.find_requirements_pkgs(project) == ["requests", "flask==2.0"]


def test_extract_env_keys_detects_getenv_and_environ(app, tmp_path):
  project = _make_project(tmp_path)
  source = (project / "main.py").read_text()
  assert app.extract_env_keys(source) == {"FOO", "BAR"}


# --- archive extraction --------------------------------------------------------

def test_extract_archive_zip_round_trip(app, tmp_path):
  project = _make_project(tmp_path / "proj")
  zpath = tmp_path / "test.zip"
  with zipfile.ZipFile(zpath, "w") as zf:
    zf.write(project / "main.py", "proj/main.py")
    zf.write(project / "requirements.txt", "proj/requirements.txt")

  dest = tmp_path / "extracted"
  app.extract_archive(zpath, dest)
  root = app.find_extraction_root(dest)

  assert root.name == "proj"
  assert (root / "main.py").exists()
  assert (root / "requirements.txt").exists()


def test_extract_archive_tar_gz_round_trip(app, tmp_path):
  project = _make_project(tmp_path / "proj")
  tpath = tmp_path / "test.tar.gz"
  with tarfile.open(tpath, "w:gz") as tf:
    tf.add(project, arcname="proj")

  dest = tmp_path / "extracted"
  app.extract_archive(tpath, dest)
  root = app.find_extraction_root(dest)

  assert root.name == "proj"
  assert (root / "main.py").exists()
  assert app.discover_python_files(root) == ["main.py", "sub/helper.py"]


def test_extract_archive_zip_rejects_path_traversal(app, tmp_path):
  evil_zip = tmp_path / "evil.zip"
  with zipfile.ZipFile(evil_zip, "w") as zf:
    zf.writestr("../../evil.py", "print('pwned')")

  dest = tmp_path / "extracted"
  with pytest.raises(ValueError):
    app.extract_archive(evil_zip, dest)


def test_extract_archive_unsupported_extension_raises(app, tmp_path):
  bogus = tmp_path / "not_an_archive.rar"
  bogus.write_text("nope")
  with pytest.raises(ValueError):
    app.extract_archive(bogus, tmp_path / "out")


def test_find_extraction_root_returns_staging_when_multiple_entries(app, tmp_path):
  (tmp_path / "a.py").write_text("x = 1\n")
  (tmp_path / "b.py").write_text("x = 2\n")
  assert app.find_extraction_root(tmp_path) == tmp_path


# --- misc text helpers ---------------------------------------------------------

def test_safe_trim_text_keeps_short_text(app):
  assert app.safe_trim_text("hello") == "hello"


def test_safe_trim_text_trims_long_text_from_the_end(app):
  text = "x" * 5000
  trimmed = app.safe_trim_text(text, limit=100)
  assert trimmed.startswith("...\n")
  assert trimmed.endswith("x" * 100)


def test_pretty_dt_from_ts_roundtrip(app):
  ts = app.now_ts_str()
  pretty = app.pretty_dt_from_ts(ts)
  assert pretty != ts  # got reformatted, not just echoed back
  assert "." in pretty and ":" in pretty


def test_pretty_dt_from_ts_passthrough_on_bad_input(app):
  assert app.pretty_dt_from_ts("not-a-timestamp") == "not-a-timestamp"
