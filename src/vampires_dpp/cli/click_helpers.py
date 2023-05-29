import click


# callback that will confirm if a flag is false
def abort_if_false(ctx, param, value):
    if not value:
        ctx.abort()


def abort_if_true(ctx, param, value):
    if value:
        ctx.abort()
