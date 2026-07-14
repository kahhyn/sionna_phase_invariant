"""Validate a channel suite and print shell-friendly domain entries."""

import argparse
import json
from pathlib import Path

from data import load_channel_profile


def load_suite(path: str | Path) -> dict:
    suite_path = Path(path)
    with suite_path.open("r", encoding="utf-8") as handle:
        suite = json.load(handle)
    if suite.get("schema_version") != 1:
        raise ValueError("channel suite schema_version must be 1")
    domains = suite.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("channel suite requires a non-empty domains list")
    seen = set()
    for domain in domains:
        domain_id = domain.get("id")
        if not domain_id or domain_id in seen:
            raise ValueError(f"invalid or duplicate suite domain id: {domain_id!r}")
        seen.add(domain_id)
        profile = domain.get("profile")
        if not profile:
            raise ValueError(f"suite domain {domain_id!r} has no profile")
        load_channel_profile(profile, component_id=domain.get("component_id"))
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    args = parser.parse_args()
    for domain in load_suite(args.suite)["domains"]:
        print(
            "\t".join(
                [
                    domain["id"],
                    domain["profile"],
                    domain.get("component_id", ""),
                ]
            )
        )


if __name__ == "__main__":
    main()
