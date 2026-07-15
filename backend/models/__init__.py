"""ORM models package for TrustLens AI backend.

Re-exports every ORM model so they can be imported directly as
``from models import User`` and, critically, so that importing this
package registers every model's table on ``database.Base.metadata`` \u2014
which ``database.init_db.init_db`` relies on to create all tables in
one pass.

To add a new model:
    1. Define it in its own module under ``models/`` (e.g.
       ``models/document.py``), inheriting from ``database.database.Base``.
    2. Import it below and add it to ``__all__``.
"""

from models.user import User

__all__ = ["User"]