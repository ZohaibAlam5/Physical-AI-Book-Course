---
id: "002"
title: "Fix RAG API Server Errors"
stage: "red"
date: "2025-12-19"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "Zohaib Alam"
command: "/sp.phr"
labels: ["rag", "api", "debugging", "fastapi", "middleware", "error-handling"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "backend/src/middleware/rate_limit.py"
  - "backend/src/api/v1/chat.py"
tests: []
---

# Fix RAG API Server Errors

## Context
User encountered errors when running the RAG chatbot API server, specifically "'dict' object is not callable" errors in the middleware and exception handlers. The server needed fixes to properly handle responses in both the rate limiting middleware and FastAPI exception handlers.

## User Input
```
ERROR:src.api.v1.chat:Internal server error: 'dict' object is not callable
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\base.py", line 78, in call_next
    message = await recv_stream.receive()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\anyio\streams\memory.py", line 98, in receive
    return self.receive_nowait()
           ~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\anyio\streams\memory.py", line 91, in receive_nowait
    raise EndOfStream
anyio.EndOfStream

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\errors.py", line 162, in __call__
    await self.app(scope, receive, _send)
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\base.py", line 108, in __call__
    response = await self.dispatch_func(request, call_next)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\Coding\GeminiClaude\book\backend\src\middleware\rate_limit.py", line 121, in __call__
    response = await call_next(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\base.py", line 84, in call_next
    raise app_exc
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\base.py", line 70, in coro
    await self.app(scope, receive_or_disconnect, send_no_error)
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\cors.py", line 83, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\exceptions.py", line 91, in __call__
    await response(scope, receive, sender)
          ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'dict' object is not callable

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\uvicorn\protocols\http\httptools_impl.py", line 426, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.scope, self.receive, self.send
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\uvicorn\middleware\proxy_headers.py", line 84, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\fastapi\applications.py", line 1106, in __call__
    await super().__call__(scope, receive, send)
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\applications.py", line 122, in __call__
    await self.middleware_stack(scope, receive, send)
  File "C:\Users\zohai\AppData\Roaming\Python\Python313\site-packages\starlette\middleware\errors.py", line 179, in __call__
    await response(scope, receive, send)
          ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'dict' object is not callable
INFO:     127.0.0.1:58731 - "GET /favicon.ico HTTP/1.1" 500 Internal Server Error
```

## Response Summary
Fixed the "'dict' object is not callable" error by updating the rate limiting middleware and FastAPI exception handlers to return proper Response objects instead of dictionaries.

## Implementation Details

### Issues Fixed:
1. Rate limiting middleware was returning HTTPException instead of JSONResponse
2. FastAPI exception handlers were returning dictionaries instead of JSONResponse objects
3. Proper ASGI response handling implemented

### Changes Made:
1. Updated rate_limit_response method in rate_limit.py to return JSONResponse
2. Updated FastAPI exception handlers in chat.py to return JSONResponse

## Outcome Impact
The API server now runs successfully without the "'dict' object is not callable" error. The server is properly handling requests and responses according to ASGI specifications.

## Files Modified
- backend/src/middleware/rate_limit.py
- backend/src/api/v1/chat.py

## Tests Summary
- Verified server starts without errors
- Confirmed API endpoints are accessible
- Validated proper error response handling

## Next Steps
1. Integrate the chatbot with the Docusaurus frontend
2. Deploy backend to cloud provider
3. Test full integration with website

## Reflection Note
Proper response object handling is crucial in ASGI applications. Always return Response objects from middleware and exception handlers, not plain dictionaries or exceptions.

## Failure Modes Observed
- Improper response object handling in middleware and exception handlers
- ASGI application expecting callable response objects

## Next Experiment to Improve Prompt Quality
N/A

## Evaluation
- Full prompt preserved verbatim: Yes
- Stage and routing determined correctly: Yes
- Metadata fields populated: Yes