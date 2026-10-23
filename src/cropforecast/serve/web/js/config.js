/* The backend a static copy of this page talks to by default.
   Empty here: the copy FastAPI serves talks to its own origin. The Netlify
   copy in frontend_deploy/ gets serve.backend_url from configs/default.yaml,
   written in by scripts/11_build_frontend.py. A ?api= in the link still wins. */
window.CROPCAST_API = "";
