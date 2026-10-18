"""funnel_gen with extra CLIProxy generators registered (2026-10-07 probe ladder; user: Composer 2.5 fast, Astra).
Registers the model on CLIProxy without editing models_v51 / trajgen_models, then runs funnel_gen.main unchanged."""
import funnel_gen
import models_v51 as routing

EXTRA = ('grok-composer-2.5-fast', 'gpt-6-astra', 'composer-2.5')
for m in EXTRA:
    routing.tm.MODELS[m] = routing.tm.CLIPROXY

if __name__ == '__main__':
    funnel_gen.main()
