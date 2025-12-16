"""program compatibility removed.

The `program` shims were removed to keep the codebase clean. Import
directly from `gui`, `backend`, or `utils` instead.
"""

raise ImportError(
	"program.* compatibility shims have been removed. "
	"Import from 'gui', 'backend', or 'utils' instead."
)

