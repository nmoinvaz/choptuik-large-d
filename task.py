#!/usr/bin/env python3
"""Tasks as items of the GitHub project "choptuik-large-d" (AGENTS.md, "Tasks: the GitHub project").

    python3 task.py list [--owner win2] [--status Todo]
    python3 task.py show N
    python3 task.py start N --by win2 "machine, checkout, threads"
    python3 task.py log N --by win2 "text" | --file FILE
    python3 task.py summary N --by win2 --file FILE
    python3 task.py status N Todo|"In Progress"|Blocked|Done
    python3 task.py done N
    python3 task.py set N [--owner win2|none] [--area A] [--estimate E] [--depends "#12, #40"] [--sprint "Sprint 1"]
    python3 task.py new "title" --file BODY [--owner ...] [--area ...] [--estimate ...] [--depends ...] [--number N]
    python3 task.py --selftest [--mutate=twolog]

A task is a project-only item (a draft item) whose title is "#N title" and whose body is the task. start, log and
summary append to the body: dated entries under "## Log", then "## Summary". A task that is still an issue of the
repository takes the same commands: its entries are posted as comments on the issue, and done also closes it.
new numbers a task one above the highest in the project, and from 1001 at least, so that a task number never meets a
number the repository gives to an issue or a pull request. Needs the gh CLI logged in with the project scope
(gh auth refresh -s project). --selftest checks the body edits offline; --mutate=twolog must make it fail.
"""
import json
import re
import subprocess
import sys
import time

OWNER, PROJECT, REPO = "nmoinvaz", 1, "nmoinvaz/choptuik-large-d"
BODY_MAX = 60000        # GitHub refuses a body above 65536 characters
FIRST_NEW = 1001
STATUSES = ("Todo", "In Progress", "Blocked", "Done")


def run(args, text=None):
    r = subprocess.run(args, input=text, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        sys.exit(f"{' '.join(args[:4])} failed: {r.stderr.strip()[:500]}")
    return r.stdout


def gql(query, **variables):
    """one GraphQL request, sent on stdin so that a long body meets no command-line limit"""
    out = json.loads(run(["gh", "api", "graphql", "--input", "-"], json.dumps(dict(query=query, variables=variables))))
    if out.get("errors"):
        sys.exit("GraphQL: " + "; ".join(e.get("message", "") for e in out["errors"])[:500])
    return out["data"]


def project():
    """the project's id and its fields by name: id, options by name, iterations by title"""
    q = """query($o: String!, $n: Int!) { user(login: $o) { projectV2(number: $n) { id fields(first: 50) { nodes {
      ... on ProjectV2FieldCommon { id name }
      ... on ProjectV2SingleSelectField { options { id name } }
      ... on ProjectV2IterationField { configuration { iterations { id title } completedIterations { id title } } } } } } } }"""
    p = gql(q, o=OWNER, n=PROJECT)["user"]["projectV2"]
    fields = {}
    for f in p["fields"]["nodes"]:
        c = f.get("configuration") or {}
        fields[f["name"]] = dict(id=f["id"], options={o["name"]: o["id"] for o in f.get("options", [])},
                                 iterations={i["title"]: i["id"] for i in c.get("iterations", []) + c.get("completedIterations", [])})
    return p["id"], fields


def number_of(title):
    """the task number of a draft item's title "#N title", or None"""
    m = re.match(r"#(\d+) ", title or "")
    return int(m.group(1)) if m else None


def items():
    """every item of the project: number, kind (draft or issue), ids, title, body, state and field values by name"""
    q = """query($o: String!, $n: Int!, $after: String) { user(login: $o) { projectV2(number: $n) {
      items(first: 100, after: $after) { pageInfo { hasNextPage endCursor } nodes { id
        content { __typename ... on DraftIssue { id title body } ... on Issue { number title body state } }
        fieldValues(first: 30) { nodes { __typename
          ... on ProjectV2ItemFieldSingleSelectValue { name field { ... on ProjectV2FieldCommon { name } } }
          ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2FieldCommon { name } } }
          ... on ProjectV2ItemFieldIterationValue { title field { ... on ProjectV2FieldCommon { name } } } } } } } } } }"""
    out, after = [], None
    while True:
        page = gql(q, o=OWNER, n=PROJECT, after=after)["user"]["projectV2"]["items"]
        for it in page["nodes"]:
            c = it.get("content") or {}
            kind = {"DraftIssue": "draft", "Issue": "issue"}.get(c.get("__typename"))
            if kind is None:
                continue
            vals = {}
            for v in it["fieldValues"]["nodes"]:
                name = (v.get("field") or {}).get("name")
                if name and name != "Title":
                    vals[name] = v.get("name") or v.get("text") or v.get("title")
            out.append(dict(item=it["id"], kind=kind, draft=c.get("id") if kind == "draft" else None,
                            number=c.get("number") if kind == "issue" else number_of(c.get("title")),
                            title=c.get("title") or "", body=c.get("body") or "", state=c.get("state"), fields=vals))
        if not page["pageInfo"]["hasNextPage"]:
            return out
        after = page["pageInfo"]["endCursor"]


def find(n, its=None):
    hit = [it for it in (its if its is not None else items()) if it["number"] == n]
    if len(hit) != 1:
        sys.exit(f"task {n}: {len(hit)} items of the project carry this number")
    return hit[0]


def stamp():
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def with_log(body, entry, mutate=None):
    """body with a dated entry appended under "## Log" (the heading is added once, before any "## Summary")"""
    if "\n## Summary" in "\n" + body:
        sys.exit("the task has its summary already; a later entry goes into a new task or a message")
    head = "" if (re.search(r"^## Log\s*$", body, re.M) and mutate != "twolog") else "\n\n## Log\n"
    return body.rstrip("\n") + (head if head else "\n") + f"\n- **{entry['when']}, {entry['by']}.** {entry['text'].strip()}\n"


def with_summary(body, entry):
    if re.search(r"^## Summary", body, re.M):
        sys.exit("the task has a summary already")
    return body.rstrip("\n") + f"\n\n## Summary ({entry['by']}, {entry['when']})\n\n{entry['text'].strip()}\n"


def checked(body):
    if len(body) > BODY_MAX:
        sys.exit(f"the body would have {len(body)} characters, above {BODY_MAX}: put the detail in the raw folder's "
                 "README.md and log one line that names it")
    return body


def put_body(it, body):
    gql("""mutation($d: ID!, $t: String!, $b: String!) { updateProjectV2DraftIssue(input: {draftIssueId: $d, title: $t,
        body: $b}) { draftIssue { id } } }""", d=it["draft"], t=it["title"], b=checked(body))


def set_field(pid, fields, item, name, value):
    """a single-select option by its name, an iteration by its title, text otherwise; None or "none" clears it"""
    f = fields[name]
    if value is None or value == "none":
        gql("""mutation($p: ID!, $i: ID!, $f: ID!) { clearProjectV2ItemFieldValue(input: {projectId: $p, itemId: $i,
            fieldId: $f}) { projectV2Item { id } } }""", p=pid, i=item, f=f["id"])
        return
    if f["options"]:
        if value not in f["options"]:
            sys.exit(f"{name}: {value!r} is not one of {', '.join(f['options'])}")
        val = dict(singleSelectOptionId=f["options"][value])
    elif f["iterations"]:
        if value not in f["iterations"]:
            sys.exit(f"{name}: {value!r} is not one of {', '.join(f['iterations'])}")
        val = dict(iterationId=f["iterations"][value])
    else:
        val = dict(text=value)
    gql("""mutation($p: ID!, $i: ID!, $f: ID!, $v: ProjectV2FieldValue!) { updateProjectV2ItemFieldValue(input: {
        projectId: $p, itemId: $i, fieldId: $f, value: $v}) { projectV2Item { id } } }""", p=pid, i=item, f=f["id"], v=val)


def post(it, kind, by, text):
    """a start, log or summary entry: into a draft item's body, or as a comment on a task that is still an issue"""
    entry = dict(when=stamp(), by=by, text=text)
    if it["kind"] == "issue":
        head = {"start": "**Started**", "log": "**Progress**", "summary": "## Closing summary"}[kind]
        run(["gh", "issue", "comment", str(it["number"]), "-R", REPO, "--body-file", "-"],
            f"{head} ({by}, {entry['when']})\n\n{text.strip()}\n")
    elif kind == "summary":
        put_body(it, with_summary(it["body"], entry))
    else:
        put_body(it, with_log(it["body"], dict(entry, text=("started. " if kind == "start" else "") + text)))


def create(title, body, number=None, its=None):
    """a new draft item "#N title"; N is one above the highest number in the project, and FIRST_NEW at least"""
    its = its if its is not None else items()
    used = {it["number"] for it in its if it["number"] is not None}
    n = number if number is not None else max([FIRST_NEW - 1] + list(used)) + 1
    if n in used:
        sys.exit(f"task {n} exists already")
    pid, fields = project()
    item = gql("""mutation($p: ID!, $t: String!, $b: String!) { addProjectV2DraftIssue(input: {projectId: $p, title: $t,
        body: $b}) { projectItem { id } } }""", p=pid, t=f"#{n} {title}", b=checked(body))["addProjectV2DraftIssue"]["projectItem"]["id"]
    set_field(pid, fields, item, "Status", "Todo")
    return n, item, pid, fields


def selftest(mutate=None):
    e1, e2 = dict(when="2026-11-10 07:00 UTC", by="win2", text="started. 18 threads"), dict(when="t2", by="win2", text="half way")
    b = with_log(with_log("**What.** A run.\n", e1, mutate), e2, mutate)
    s = with_summary(b, dict(when="t3", by="win2", text="Done, exit 0."))
    ok = [number_of("#293 Count D 58.01's boxes") == 293, number_of("293 no hash") is None, number_of("Count #5") is None,
          b.count("## Log") == 1, b.index("started. 18 threads") < b.index("half way"), b.startswith("**What.** A run.\n\n## Log\n"),
          s.index("## Log") < s.index("## Summary (win2, t3)"), s.endswith("Done, exit 0.\n")]
    print(f"task.py selftest: {sum(ok)} of {len(ok)} checks pass" + (f" (mutation {mutate})" if mutate else ""))
    return 0 if all(ok) else 1


def main(argv):
    opt, args, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" in a:
            opt[a[2:].split("=", 1)[0]] = a.split("=", 1)[1]
        elif a.startswith("--") and a != "--selftest":
            opt[a[2:]] = argv[i + 1]
            i += 1
        else:
            args.append(a)
        i += 1
    if "--selftest" in args:
        return selftest(opt.get("mutate"))
    if not args:
        sys.exit(__doc__)
    cmd = args[0]
    text = open(opt["file"], encoding="utf-8").read() if "file" in opt else (args[2] if len(args) > 2 else "")
    meta = dict(owner="Owner", area="Area", estimate="Estimate", depends="Depends on", sprint="Sprint")
    if cmd == "list":
        for it in sorted(items(), key=lambda it: it["number"] or 0):
            f = it["fields"]
            if opt.get("owner", f.get("Owner")) == f.get("Owner") and opt.get("status", f.get("Status")) == f.get("Status"):
                print(f"{it['number']}\t{f.get('Status') or '-'}\t{f.get('Owner') or '-'}\t{it['kind']}\t{re.sub(r'^#[0-9]+ ', '', it['title'])}")
        return 0
    if cmd == "new":
        n, item, pid, fields = create(args[1], text, int(opt["number"]) if "number" in opt else None)
        for k, name in meta.items():
            if k in opt:
                set_field(pid, fields, item, name, opt[k])
        print(f"task {n} created")
        return 0
    it = find(int(args[1].lstrip("#")))
    if cmd == "show":
        print(f"#{it['number']} {re.sub(r'^#[0-9]+ ', '', it['title'])}  [{it['kind']}{', ' + it['state'].lower() if it['state'] else ''}]")
        print("  ".join(f"{k}: {v}" for k, v in it["fields"].items()) + "\n")
        print(it["body"])
        return 0
    if cmd in ("start", "log", "summary"):
        if "by" not in opt or not text.strip():
            sys.exit(f"{cmd} needs --by <name> and a text or --file")
        post(it, cmd, opt["by"], text)
    pid, fields = project()
    if cmd in ("start", "done", "status"):
        value = {"start": "In Progress", "done": "Done"}.get(cmd) or args[2]
        if value not in STATUSES:
            sys.exit(f"status is one of {', '.join(STATUSES)}")
        set_field(pid, fields, it["item"], "Status", value)
        if cmd == "done" and it["kind"] == "issue" and it["state"] == "OPEN":
            run(["gh", "issue", "close", str(it["number"]), "-R", REPO])
    elif cmd == "set":
        for k, name in meta.items():
            if k in opt:
                set_field(pid, fields, it["item"], name, opt[k])
    elif cmd not in ("log", "summary"):
        sys.exit(__doc__)
    print(f"task {it['number']}: {cmd} done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
