"""CLI for local human-reviewed AVAMALE workflow prototypes."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys

from avamale_review import (InputError, load_transactions, reconcile, load_freight,
                            freight_review, read_table, write_report)
from .workflows import SPEC, SENSITIVE_SYNTHETIC_ONLY, run_workflow
from .feedback import add_feedback, summarize, delete_feedback, set_feedback_status, CATEGORIES, STATES
from .addons import ADDONS, run_addon


def run_one(code, args):
    if code not in SPEC:
        raise InputError('Unrecognized workflow: '+code)
    if code in SENSITIVE_SYNTHETIC_ONLY and not args.synthetic:
        raise InputError(f'{code} is intentionally synthetic-only until formal risk/privacy/clinical/compliance controls are approved')
    paths=[]
    if code == 'AA02':
        if not args.bank or not args.ledger: raise InputError('AA02 requires --bank and --ledger')
        bank,ledger=load_transactions(args.bank),load_transactions(args.ledger)
        result=reconcile(bank,ledger,args.date_tolerance)
        paths=[args.bank,args.ledger]
        rules={'engine':'AA02 reviewed prototype v0.1.0','tolerance_days':args.date_tolerance,
               'match':'unique exact amount/currency/reference within explicit date window',
               'limitations':'No split payments, postings, complete balance proof or integrations'}
    elif code == 'AA03':
        if not args.data:raise InputError('AA03 requires --data')
        records=load_freight(args.data)
        result=freight_review(records)
        paths=[args.data]
        rules={'engine':'AA03 reviewed prototype v0.1.0','comparison':'pre-transcribed item-level records only',
               'limitations':'No OCR, customs decision, external authenticity check or shipment release'}
    else:
        if not args.data:raise InputError(f'{code} requires --data')
        fields=SPEC[code][1].split(',')
        data=read_table(args.data,fields)
        as_of=date.fromisoformat(args.as_of)
        result=run_workflow(code,data,as_of)
        paths=[args.data]
        rules={'engine':'Rule-based review foundation 0.2.0','workflow':code,
               'as_of':str(as_of),'input_columns':fields,
               'no_external_calls':True,'no_automated_action':True,
               'review_required':True,
               'scope':'Evidence metadata/structured-input analysis only; requires buyer-specific schema and acceptance tests'}
    manifest=write_report(result,args.out,f'{code} · {SPEC[code][0]}',paths,rules,args.synthetic)
    return {'workflow':code,'out':str(Path(args.out).resolve()),'counts':manifest['record_counts'],
            'test_only':args.synthetic,'human_review_required':True,'production_certified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='List supported workflows and schema')
    run=sub.add_parser('run',help='Run a local, non-destructive review')
    run.add_argument('workflow',choices=sorted(SPEC))
    run.add_argument('--data',help='CSV or values-only XLSX for AA01/AA03/AA04..AA18')
    run.add_argument('--bank',help='AA02 bank file')
    run.add_argument('--ledger',help='AA02 ledger file')
    run.add_argument('--date-tolerance',type=int,default=0)
    run.add_argument('--out',required=True)
    run.add_argument('--as-of',default='2026-11-07',help='Date for expiry checks YYYY-MM-DD')
    run.add_argument('--synthetic',action='store_true',help='Operator asserts data are wholly synthetic')
    addon=sub.add_parser('addon',help='Synthetic-only product add-ons: Biz Assist, AskDocs and enquiry intake')
    addon_sub=addon.add_subparsers(dest='addon_command',required=True)
    addon_sub.add_parser('list')
    addon_run=addon_sub.add_parser('run')
    addon_run.add_argument('product',choices=sorted(ADDONS))
    addon_run.add_argument('--data',required=True,help='Synthetic CSV input')
    addon_run.add_argument('--question',default='',help='AskDocs retrieval query')
    addon_run.add_argument('--synthetic',action='store_true',help='Operator asserts entirely synthetic example inputs')
    addon_run.add_argument('--out',required=True,help='New result JSON path, refusing overwrites')
    demo=sub.add_parser('demo',help='Run all 18 supplied synthetic fixtures and generate reports')
    demo.add_argument('--out',required=True)
    fb=sub.add_parser('feedback',help='Local consent-gated de-identified product feedback')
    fbsub=fb.add_subparsers(dest='feedback_command',required=True)
    add=fbsub.add_parser('add')
    add.add_argument('--file',required=True);add.add_argument('--workflow',required=True,choices=sorted(set(SPEC)|set(ADDONS)))
    add.add_argument('--category',required=True,choices=sorted(CATEGORIES));add.add_argument('--summary',required=True)
    add.add_argument('--impact',required=True,choices=['low','medium','high'])
    add.add_argument('--source-type',required=True,choices=['consented_interview','approved_deidentified_summary','synthetic_test','support_ticket_optin'])
    add.add_argument('--source-hint',default='');add.add_argument('--consent-confirmed',action='store_true')
    summary=fbsub.add_parser('summary');summary.add_argument('--file',required=True)
    delete=fbsub.add_parser('delete');delete.add_argument('--file',required=True);delete.add_argument('--id',required=True)
    review=fbsub.add_parser('review');review.add_argument('--file',required=True);review.add_argument('--id',required=True)
    review.add_argument('--status',required=True,choices=sorted(STATES));review.add_argument('--reviewed',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='list': print(json.dumps({k:{'title':v[0],'input_columns':v[1],'synthetic_only':k in SENSITIVE_SYNTHETIC_ONLY} for k,v in SPEC.items()},indent=2));return 0
        if args.command=='run': print(json.dumps(run_one(args.workflow,args),indent=2));return 0
        if args.command=='addon':
            if args.addon_command=='list':
                print(json.dumps({k:{'title':v[0],'csv_schema':v[1],'synthetic_only':True,'human_review':True} for k,v in ADDONS.items()},indent=2));return 0
            if not args.synthetic:
                raise InputError('Add-on demos accept synthetic inputs only. Never upload real customer or transaction data.')
            file=Path(args.data)
            if file.stat().st_size > 2_000_000: raise InputError('Demo CSV size must not exceed 2MB')
            fields=ADDONS[args.product][1].split(',')
            with file.open(newline='',encoding='utf-8-sig') as handle:
                reader=csv.DictReader(handle)
                if reader.fieldnames is None or any(f not in reader.fieldnames for f in fields):
                    raise InputError('Missing CSV headers: '+','.join(f for f in fields if reader.fieldnames is None or f not in reader.fieldnames))
                rows=list(reader)
            matches=run_addon(args.product,rows,args.question)
            target=Path(args.out)
            if target.exists(): raise InputError('Refusing to overwrite existing review output')
            target.parent.mkdir(parents=True,exist_ok=True)
            payload={'product':args.product,'production_certified':False,'synthetic_only':True,'human_review_required':True,'no_automated_actions':True,'results':matches}
            with target.open('x',encoding='utf-8') as out: json.dump(payload,out,indent=2)
            print(json.dumps({'product':args.product,'saved_to':str(target.resolve()),'findings':len(matches)},indent=2));return 0
        if args.command=='demo':
            root=Path(__file__).resolve().parents[1];out=Path(args.out)
            output=[]
            for code in SPEC:
                params=['run',code,'--out',str(out/code),'--synthetic','--as-of','2026-11-07']
                if code=='AA02': params+=['--bank',str(root/'samples/bank_synthetic.csv'),'--ledger',str(root/'samples/ledger_synthetic.csv'),'--date-tolerance','1']
                else:params+=['--data',str(root/'samples'/f'{code}.csv') if code!='AA03' else str(root/'samples/freight_synthetic.csv')]
                # Pass through the same argument parser so demo validates the public command surface.
                from io import StringIO
                from contextlib import redirect_stdout
                with redirect_stdout(StringIO()) as stdout:main(params)
                output.append(json.loads(stdout.getvalue()))
            print(json.dumps({'generated_reports':len(output),'outputs':output},indent=2));return 0
        if args.command=='feedback':
            if args.feedback_command=='add':
                value=add_feedback(args.file,args.workflow,args.category,args.summary,args.impact,args.source_type,args.consent_confirmed,args.source_hint)
            elif args.feedback_command=='summary':value=summarize(args.file)
            elif args.feedback_command=='review':value=set_feedback_status(args.file,args.id,args.status,args.reviewed)
            else:value={'deleted':delete_feedback(args.file,args.id)}
            print(json.dumps(value,indent=2));return 0
    except (InputError,ValueError,OSError) as exc:
        parser.exit(2,f'Validation error: {exc}\n')

if __name__=='__main__':
    sys.exit(main())
