<div align="center">
<br>
<img src="https://raw.githubusercontent.com/anbasile/tango/main/docs/source/_static/tango_final_horizontal.png" width="600"/>
<br>
<br>
<p>
<!-- start tagline -->
AI2 Tango replaces messy directories and spreadsheets full of file versions by organizing experiments into discrete steps that can be cached and reused throughout the lifetime of a research project.
<!-- end tagline -->
</p>
<hr/>
<a href="https://github.com/anbasile/tango/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/anbasile/tango/actions/workflows/ci.yml/badge.svg?branch=main">
</a>
<a href="https://github.com/anbasile/tango/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/anbasile/tango?display_name=tag&sort=semver">
</a>
<a href="https://github.com/anbasile/tango/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/anbasile/tango.svg?color=blue&cachedrop">
</a>
<br/>
</div>

> ### About this fork
>
> This is an unofficial fork of [allenai/tango](https://github.com/allenai/tango), which has been
> unmaintained since May 2024 (last release v1.3.2, October 2023). It updates Tango to run on a
> current Python and ML stack — Python 3.10–3.13, PyTorch 2.x, transformers 5.x, datasets 5.x —
> and drops the integrations whose upstreams are gone. See [the CHANGELOG](CHANGELOG.md) for the
> full list of changes.
>
> It is **not** published to PyPI or conda-forge, and it is not affiliated with AI2. The original
> `ai2-tango` package on PyPI is the upstream project, not this one.

## Quick links

- [Releases](https://github.com/anbasile/tango/releases)
- [CHANGELOG](CHANGELOG.md)
- [Contributing](.github/CONTRIBUTING.md)
- [License](LICENSE)

## In this README

- [Quick start](#quick-start)
- [Installation](#installation)
  - [Installing with PIP](#installing-with-pip)
  - [Installing with Conda](#installing-with-conda)
  - [Installing from source](#installing-from-source)
  - [Checking your installation](#checking-your-installation)
  - [Docker image](#docker-image)
- [FAQ](#faq)
- [Team](#team)
- [License](#license)

## Quick start

Create a Tango step:

```python
# hello.py

from tango import step

@step()
def hello(name: str) -> str:
    message = f"Hello, {name}!"
    print(message)
    return message
```

And create a corresponding experiment configuration file:

```jsonnet
// hello.jsonnet

{
  steps: {
    hello: {
      type: "hello",
      name: "World",
    }
  }
}
```

Then run the experiment using a local workspace to cache the result:

```bash
tango run hello.jsonnet -w /tmp/workspace
```

You'll see something like this in the output:

```
Starting new run expert-llama
● Starting step "hello"...
Hello, World!
✓ Finished step "hello"
✓ Finished run expert-llama
```

If you run this a second time the output will now look like this:

```
Starting new run open-crab
✓ Found output for step "hello" in cache...
✓ Finished run open-crab
```

You won't see "Hello, World!" this time because the result of the step was found in the cache, so it wasn't run again.

For a more detailed introduction check out the [First Steps](https://ai2-tango.readthedocs.io/en/latest/first_steps.html) walk-through.

## Installation

<!-- start install -->

This fork requires Python 3.10 or later.

### Installing a release

This fork is not on PyPI. Wheels are attached to
[each GitHub release](https://github.com/anbasile/tango/releases), so install one directly:

```bash
pip install https://github.com/anbasile/tango/releases/download/v2.0.0/ai2_tango-2.0.0-py3-none-any.whl
```

Extras work as usual:

```bash
pip install 'ai2_tango[torch] @ https://github.com/anbasile/tango/releases/download/v2.0.0/ai2_tango-2.0.0-py3-none-any.whl'
```

The available extras are `torch`, `transformers`, `datasets`, `examples` and `all`.

### Installing from source

To install **ai2-tango** from source, first clone [the repository](https://github.com/anbasile/tango):

```bash
git clone https://github.com/anbasile/tango.git
cd tango
```

Then run

```bash
pip install -e '.[all]'
```

To install with only a specific integration, such as `torch` for example, run

```bash
pip install -e '.[torch]'
```

Or to install just the base tango library, you can run

```bash
pip install -e .
```

### Checking your installation

Run

```bash
tango info
```

to check your installation.

### Docker image

No prebuilt images are published for this fork, but [the Dockerfile](Dockerfile) builds one:

```bash
docker build -t tango .
docker run --rm tango info
```

It defaults to a CUDA-enabled `pytorch/pytorch` base, which you can override:

```bash
docker build --build-arg BASE_IMAGE=python:3.12-slim -t tango .
```

<!-- end install -->

## FAQ

<!-- start faq -->

### Why is the library named Tango?

The motivation behind this library is that we can make research easier by composing it into well-defined steps.  What happens when you choreograph a number of steps together?  Well, you get a dance.  And since our [team's leader](https://nasmith.github.io/) is part of a tango band, "AI2 Tango" was an obvious choice!

### How can I debug my steps through the Tango CLI?

You can run the `tango` command through [pdb](https://docs.python.org/3/library/pdb.html). For example:

```bash
python -m pdb -m tango run config.jsonnet
```

### How is Tango different from [Metaflow](https://metaflow.org), [Airflow](https://airflow.apache.org), or [redun](https://github.com/insitro/redun)?

We've found that existing DAG execution engines like these tools are great for production workflows but not as well suited for messy, collaborative research projects
where code is changing constantly. AI2 Tango was built *specifically* for these kinds of research projects.

### How does Tango's caching mechanism work?

AI2 Tango caches the results of steps based on the `unique_id` of the step. The `unique_id` is essentially a hash of all of the inputs to the step along with:

1. the step class's fully qualified name, and
2. the step class's `VERSION` class variable (an arbitrary string).

Unlike other workflow engines like [redun](https://github.com/insitro/redun), Tango does *not* take into account the source code of the class itself (other than its fully qualified name) because we've found that using a hash of the source code bytes is way too sensitive and less transparent for users.
When you change the source code of your step in a meaningful way you can just manually change the `VERSION` class variable to indicate to Tango
that the step has been updated.

<!-- end faq -->

## Team

<!-- start team -->

**ai2-tango** was created and maintained by the AllenNLP team, backed by
[the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/), and the overwhelming
majority of this codebase is their work — see
[the upstream contributors](https://github.com/allenai/tango/graphs/contributors).

This fork is maintained separately by [@anbasile](https://github.com/anbasile) and is not
affiliated with or endorsed by AI2.

<!-- end team -->

## License

<!-- start license -->

**ai2-tango** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/anbasile/tango/blob/main/LICENSE).

<!-- end license -->
