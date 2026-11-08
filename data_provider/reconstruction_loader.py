"""Same-window ALFA reconstruction data, retaining prepared flight boundaries."""

import json
from pathlib import Path

import numpy as np

from data.alfa import FEATURES, FlightWindows, load_prepared
from data.alfa_within_flight import PROTOCOL


class ALFAReconstructionDataset(FlightWindows):
    """Return (window, flight_index, start), without modifying values or labels."""

    def __init__(self, args, flag, flights=None):
        if flag not in {'train', 'val', 'test'}:
            raise ValueError(f'Unknown split: {flag}')
        root = Path(args.root_path)
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('protocol') != PROTOCOL:
            raise ValueError('Reconstruction requires prepared within-flight ALFA data, not injection CSVs')
        if self.manifest['features'] != list(FEATURES):
            raise ValueError('Prepared feature order differs from ALFA reconstruction features')
        if args.seq_len < 2 or args.seq_len % 2:
            raise ValueError('WTFTP reconstruction requires an even seq_len >= 2')
        if args.seq_len != self.manifest['windowing']['expected_window']:
            raise ValueError('seq_len differs from the prepared dataset window')
        if args.enc_in != len(FEATURES) or args.c_out != len(FEATURES):
            raise ValueError('Reconstruction requires enc_in = c_out = 18 (including roll)')
        self.all_flights = load_prepared(root) if flights is None else flights
        self.columns = list(FEATURES)
        self.normalization = self.manifest['normalization']
        selected = [flight for flight in self.all_flights if flight.split == flag]
        if not selected:
            raise ValueError(f'No prepared flights for {flag}')
        for flight in selected:
            if (flight.values.shape != (len(flight.labels), len(FEATURES)) or
                    not np.isfinite(flight.values).all() or
                    not np.isin(flight.labels, [0, 1]).all()):
                raise ValueError(f'Invalid prepared values or labels: {flight.name}')
            if flag != 'test' and flight.labels.any():
                raise ValueError(f'{flag} must contain normal points only: {flight.name}')
        super().__init__(self.all_flights, flag, window=args.seq_len)
