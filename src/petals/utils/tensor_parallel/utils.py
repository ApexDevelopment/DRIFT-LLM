"""
Helpers for processing nested dicts, tuples, lists and namedtuples.

Vendored (and trimmed) from the ``tensor_parallel`` library, which is no longer maintained.
See ``petals/utils/tensor_parallel/__init__.py`` for the rationale.
"""


def nested_compare(t, u):
	"""Return whether the nested structure of ``t`` and ``u`` matches."""
	if isinstance(t, (list, tuple)):
		if not isinstance(u, type(t)):
			return False
		if len(t) != len(u):
			return False
		return all(nested_compare(a, b) for a, b in zip(t, u))
	if isinstance(t, dict):
		if not isinstance(u, dict):
			return False
		if set(t.keys()) != set(u.keys()):
			return False
		return all(nested_compare(t[k], u[k]) for k in t)
	return True


def nested_flatten(t):
	"""Turn a nested list/tuple/dict into a flat iterator."""
	if isinstance(t, (list, tuple)):
		for x in t:
			yield from nested_flatten(x)
	elif isinstance(t, dict):
		for k, v in sorted(t.items()):
			yield from nested_flatten(v)
	else:
		yield t


def nested_pack(flat, structure):
	"""Restore a nested structure from a flat iterable, using ``structure`` as a template."""
	return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
	if is_namedtuple(structure):
		return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
	elif isinstance(structure, (list, tuple)):
		return type(structure)(_nested_pack(flat_iter, x) for x in structure)
	elif isinstance(structure, dict):
		return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
	else:
		return next(flat_iter)


def is_namedtuple(x):
	"""Check whether ``x`` is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
	t = type(x)
	b = t.__bases__
	if len(b) != 1 or b[0] != tuple:
		return False
	f = getattr(t, "_fields", None)
	if not isinstance(f, tuple):
		return False
	return all(type(n) == str for n in f)
