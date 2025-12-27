import json
import datasets
from fire import Fire
from functools import partial
from typing import List
from loguru import logger
import asyncio

from utils import (
    generate_together,
    generate_openai,
    generate_with_references,
    generate_with_references_async,
    DEBUG,
)


async def process_fn_async(
    item,
    model,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
):
    """Async version that parallelizes reference model calls."""
    messages = [{"role": "user", "content": item["instruction"]}]
    references = item.get("references", [])

    if len(references) == 0 and len(reference_models) > 0:
        prev_references = []

        for i_round in range(rounds):
            if DEBUG:
                logger.info(
                    f"Round {i_round+1}/{rounds} to collecting reference responses."
                )

            # KEY CHANGE: Parallelize reference model calls using asyncio.gather
            tasks = [
                generate_with_references_async(
                    model=reference_model,
                    messages=messages,
                    references=prev_references,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                for reference_model in reference_models
            ]

            # Wait for all reference models to complete in parallel
            reference_results = await asyncio.gather(*tasks)

            # Filter out None results
            references = [ref for ref in reference_results if ref is not None]

            if i_round < rounds - 1:
                prev_references = references
                references = []

    # Generate final output with aggregator model
    output = await generate_with_references_async(
        model=model,
        messages=messages,
        references=references,
    )

    return {"output": output, "generator": model + "-together"}


def process_fn(
    item,
    model,
    reference_models=[],
    temperature=0.7,
    max_tokens=2048,
    rounds=1,
):
    """Synchronous wrapper that calls async version."""
    # Run the async function in a new event loop
    return asyncio.run(
        process_fn_async(
            item=item,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
        )
    )


def main(
    model: str,
    output_path: str,
    reference_paths: str = None,
    reference_models: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    num_proc: int = 16,
):

    if reference_paths is None:
        reference_paths = []
    else:
        reference_paths = reference_paths.split(",")

    if reference_models is None:
        reference_models = []
    else:
        reference_models = reference_models.split(",")

    eval_set = datasets.load_dataset(
        "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline", trust_remote_code=True
    )["eval"]
    eval_set = eval_set.remove_columns(["output", "generator"])

    if len(reference_paths):

        logger.info(f"`reference_paths` provided: {reference_paths}")

        references = []
        for reference_path in reference_paths:
            with open(reference_path) as f:
                reference_responses = json.load(f)
                logger.info(
                    f"Reading reference outputs: {reference_path} ({len(reference_responses)})"
                )
                for i_reference_response, reference_response in enumerate(
                    reference_responses
                ):
                    if len(references) <= i_reference_response:
                        references.append([reference_response["output"]])
                    else:
                        references[i_reference_response].append(
                            reference_response["output"]
                        )

        eval_set = eval_set.add_column(f"references", references)

    elif len(reference_models):

        logger.info(
            f"`reference_models` provided: {reference_models}. Will generate reference responses on-the-fly."
        )

    logger.info(f"Start.")

    eval_set = eval_set.map(
        partial(
            process_fn,
            model=model,
            reference_models=reference_models,
            temperature=temperature,
            max_tokens=max_tokens,
            rounds=rounds,
        ),
        batched=False,
        num_proc=num_proc,
    )

    logger.info(f"Saving outputs to {output_path}.")

    try:
        eval_set = eval_set.remove_columns(f"references")
    except Exception as e:
        pass

    with open(output_path, "w") as f:

        json.dump(list(eval_set), f, indent=2)


if __name__ == "__main__":

    Fire(main)
