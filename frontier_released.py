"""Run a frontier model single-shot on the OFFICIAL released oolong-synth rows
(their exact context_window_text + question + gold) at <=8K context, graded with
our OOLONG-official graders. Compares directly to the paper's ~0.85 at 8K."""
import ast, asyncio, os, collections
import litellm, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import tasks

litellm.drop_params = True
MODEL = os.environ.get("FRONTIER", "gpt-5.4")
N = int(os.environ.get("N", "60"))
SEM = asyncio.Semaphore(int(os.environ.get("CONC", "6")))
MODE = {"ANSWER_TYPE.NUMERIC": "numeric", "ANSWER_TYPE.COMPARISON": "oolong_compare"}

def parse_gold(s):
    try:
        v = ast.literal_eval(s)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)], False
    except Exception:
        return [s], True   # datetime repr etc. -> broken gold

async def one(row):
    prompt = f"{row['context_window_text']}\n\n{row['question']}"
    async with SEM:
        try:
            r = await litellm.acompletion(model=MODEL,
                messages=[{"role": "user", "content": prompt}], temperature=0)
            txt = r.choices[0].message.content or ""
        except Exception as e:
            txt = f"(error: {e})"
    gold, broken = parse_gold(row["answer"])
    mode = MODE.get(row["answer_type"], "oolong_exact")
    sc = tasks.grade_answer(txt, gold, mode)
    return row["answer_type"], row["task"], sc, broken

async def main():
    fp = hf_hub_download("oolongbench/oolong-synth",
                         "data/validation-00000-of-00007.parquet", repo_type="dataset")
    df = pq.read_table(fp, columns=["context_len","task","answer_type","question",
                                    "answer","context_window_text"]).to_pandas()
    df = df[df.context_len <= 8192].head(N)
    print(f"RELEASED oolong-synth | model={MODEL} | n={len(df)} (<=8K context)")
    res = await asyncio.gather(*[one(r) for _, r in df.iterrows()])
    by_at = collections.defaultdict(list)
    for at, tk, sc, broken in res:
        by_at[at].append(sc)
    overall = sum(s for _,_,s,_ in res)/len(res)
    nbroken = sum(1 for *_,b in res if b)
    print(f"\nOVERALL: {overall:.3f}  (n={len(res)}, broken-gold rows={nbroken})")
    print("by answer_type:")
    for at, ss in sorted(by_at.items()):
        print(f"   {at:28s} {sum(ss)/len(ss):.3f}  (n={len(ss)})")

asyncio.run(main())
