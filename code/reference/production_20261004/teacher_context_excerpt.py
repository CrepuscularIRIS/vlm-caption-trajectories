"""Reading excerpt of caption_tiered.py: GUIDANCE and TeacherContext only.
Not a standalone generator; see README.md in this directory.
"""

GUIDANCE = ('Record what the current evidence shows, which instance it belongs to, and what remains uncertain. '
            'If candidates remain unresolved, inspect a distinguishing feature. Preserve a supported instance-attribute '
            'binding when checking other evidence. Explicitly revise it if new evidence changes it, and reconsider dependent '
            'conclusions. If a crop lacks the needed context, change the region. Answer directly when the original image is '
            'sufficient. Do not invent an error, unnecessary crop, HOLD or REVISE to match a demonstration.')

class TeacherContext:
    """Scaffold stays outside target transcript/state and the exported SFT labels."""
    def __init__(self, api, example, target_id):
        self.api, self.example, self.target_id = api, example, target_id
        self.format_used = False

    def image(self, path): return self.api.image(path)

    def context(self):
        e = self.example
        def rename(s):
            import re
            return re.sub(r'\b(original_image|observation_\d+)\b', r'demo_\1', s)
        blocks = [dict(type='text', text='COMPLETE DEMONSTRATION — a different completed episode. Demo view names are illustrative only; they cannot be used for the target.\nQuestion: ' + e['question'])]
        by_id = {v['view_id']: v for v in e['views']}
        for step in e['steps']:
            before = 'original_image' if step['step'] == 1 else e['steps'][step['step'] - 2]['observation']
            if before:
                blocks.extend([dict(type='text', text='Received [' + rename(before) + ']'), self.api.image(by_id[before]['image_path'])])
            blocks.append(dict(type='text', text='Demonstration assistant:\n' + rename(step['raw_text'])))
        blocks.append(dict(type='text', text='END DEMONSTRATION. Start a NEW target episode at Step 1. Only target original_image is initially available; all demo images are unavailable for target actions.\n' + GUIDANCE))
        return dict(role='user', content=blocks)

    def call(self, label, msgs, *args, **kwargs):
        if '/fmt' in label:
            if self.format_used:
                return dict(content='', finish_reason='error', errors=[{'error_type': 'trajectory_format_retry_cap'}])
            self.format_used = True
        # Acknowledgement closes the example turn; target messages follow unchanged.
        context = [msgs[0], self.context(), dict(role='assistant', content='The demonstration is complete. I will investigate the new target independently.')]
        return self.api.call(label, context + msgs[1:], *args, **kwargs)
