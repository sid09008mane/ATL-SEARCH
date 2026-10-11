import json,sys,re,datetime,collections
raw={x['id']:x for x in json.load(open(sys.argv[1]))}
orig=json.load(open(sys.argv[2]))
outdir=sys.argv[3]
ORDER=[int(r['name'].split('.')[0]) for r in orig]
CAT={int(r['name'].split('.')[0]):r['verified_category'] for r in orig}
NAMES={15862952:"Thota Sridhar Reddy & Ors. v. Mandala Ramulamma & Ors.",290532:"State of Haryana v. Mukesh Kumar & Ors.",702009:"Hemaji Waghaji Jat v. Bhikhabhai Khengarbhai Harijan & Ors.",199096823:"Ravinder Kaur Grewal & Ors. v. Manjit Kaur & Ors.",621206:"Manekji Edulji Mistry & Ors. v. Maneksha Ardeshir Irani & Anr.",112778127:"Tribhuvanshankar v. Amrutlal",122806905:"Uttam Chand (D) through LRs. v. Nathu Ram (D) through LRs. & Ors.",974908:"Sidram Narsappa Kamble v. Sholapur Borough Municipality & Anr.",422905:"D.N. Venkatarayappa & Anr. v. State of Karnataka & Ors.",1944776:"Koshal Pal & Ors. v. Mohan Lal & Ors.",1219878:"Vithal Dattatraya Kulkarni & Ors. v. Shamrao Tukaram Power & Ors.",47115781:"Bangalore Development Authority v. N. Jayamma",319321:"Sheikh Noor & Anr. v. Sheikh G.S. Ibrahim (Dead) by LRs.",10099:"Manik Lal Mazumdar & Ors. v. Gouranga Chandra Dey & Ors.",1571903:"Chatti Konati Rao & Ors. v. Palle Venkata Subba Rao",1633759:"Chinnamma v. Gopalan & Ors.",1452565:"Edukanti Kistamma (Dead) through LRs. & Ors. v. S. Venkatareddy (Dead) through LRs. & Ors.",1586855:"Ajit Chopra v. Sadhu Ram & Ors.",77103725:"Hindustan Petroleum Corporation Ltd. v. Dilbahar Singh",1308176:"Jamnalal & Ors. v. Radheshyam"}
CATMAP={'land_property_dispute':'property','tenant_dispute':'tenant','both':'property_and_tenant'}
W=lambda s:len(s.split())

def clean(t,old):
    t=t.replace('\x0c','\n').replace(' ',' ').replace('\t',' ')
    t=t.replace('\xad','-')
    t=t.replace('#','')
    t=re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]','',t)
    t=re.sub(r'(?m)^[ ]*\d(?:[ ]{0,2}\d){0,4}[ ]*$','',t)          # page-number-only lines
    if old: t=re.sub(r'(?<=[a-z])-{1,2}[ ]*\n[ ]*(?=[a-z])','',t)   # wrapped words in typeset reports
    else:   t=re.sub(r'(?<=\w)-[ ]*\n[ ]*(?=\w)','-',t)
    t=re.sub(r'\s+',' ',t).strip()
    t=re.sub(r'\s+([,;])',r'\1',t)
    t=re.sub(r'([….]){6,}','',t).strip()
    return t

SIG=re.compile(r'(\.{3,}|…+)\s*,?\s*(C\.?\s*)?J\b|^\W*\(?[A-Z][A-Za-z.\s]{2,40}\)?\W*$')
def is_trailer(t):
    if W(t)>40: return False
    if re.search(r'(\.{2,}|…)\s*\.?,?\s*(C\.?)?J\.?',t): return True
    if re.fullmatch(r'[\W]*(New Delhi)?[\s,;.]*([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}(st|nd|rd|th)?\s+[A-Z][a-z]+,?\s+\d{4})?[\s.,;]*',t,re.I) : return True
    if re.fullmatch(r'([A-Z]\.){2,4}',t.replace(' ','')): return True         # reporter initials
    if W(t)<=14 and re.search(r'New Delhi',t,re.I): return True
    if W(t)<=10 and re.fullmatch(r'[\W]*\(?[A-Z][A-Za-z.\s]+\)?[\W]*(J\.?|JJ\.?)?[\W]*',t) and not re.search(r'dismiss|allow|order|cost',t,re.I): return True
    return False

ABBR=re.compile(r"(?:\b(?:v|vs|No|Nos|no|nos|s|ss|Rs|Dr|Mr|Mrs|Ms|Smt|Shri|Sri|Ltd|Co|Anr|Ors|ors|anr|JJ|J|C\.J|i\.e|e\.g|p|pp|para|paras|cl|Art|Sec|Secs|viz|etc|Ex|Exh|Vol|vol|ibid|cf|St|Thr|thr|Lrs|LRs|Hon|M/s|Govt|Regd|Mst|Sch|u/s|w\.e\.f|Ch|Pt|Addl|Asst|Jr|Sr|ILR|AIR|SCR|SCC|A\.I\.R|S\.C\.R|S\.C\.C|Corpn|Bros|Devt|Auth|Prop)|\b[A-Z]|\b\d{1,3})\.$")
def sentences(t):
    parts=re.split(r'(?:(?<=[.?!])|(?<=[.?!]["”\')]))\s+(?=["“\'(\[]?[A-Z0-9])',t)
    out=[]
    for p in parts:
        if out and (ABBR.search(out[-1]) or W(out[-1])<4): out[-1]+=' '+p
        else: out.append(p)
    fin=[]
    for p in out:
        if W(p)<=250: fin.append(p); continue
        sub=re.split(r'(?<=[;:])\s+(?=\(?[a-z0-9]{1,4}\)\s|[A-Z"“(])',p); acc=''
        for q in sub:
            if acc and W(acc)+W(q)>200: fin.append(acc); acc=q
            else: acc=(acc+' '+q).strip()
        if acc: fin.append(acc)
    out=[]
    for p in fin:
        ws=p.split()
        while len(ws)>330: out.append(' '.join(ws[:220])); ws=ws[220:]
        out.append(' '.join(ws))
    return out
def pack(t,target=220,hard=320,overlap=True):
    if W(t)<=hard: return [t]
    ss=sentences(t); chunks=[];cur=[]
    for s in ss:
        if cur and W(' '.join(cur))+W(s)>target:
            chunks.append(cur); cur=[cur[-1]] if overlap and W(cur[-1])<=60 else []
        cur.append(s)
        while W(' '.join(cur))>hard*1.6 and len(cur)==1:      # one giant "sentence": cut on words
            ws=cur[0].split(); chunks.append([' '.join(ws[:target])]); cur=[' '.join(ws[target:])]
    if cur: chunks.append(cur)
    res=[' '.join(c) for c in chunks]
    if len(res)>1 and W(res[-1])<50: res[-2:]=[res[-2]+' '+ (res[-1] if not overlap else ' '.join(chunks[-1][1:]) or res[-1])]
    return res

def best_chain(cands,maxstep=3):
    """cands: list of (number, weight). Longest increasing run (steps 1..maxstep, start<=3)."""
    if not cands: return []
    best=[0.0]*len(cands);prev=[-1]*len(cands)
    for i,(n,w) in enumerate(cands):
        best[i]=w if n<=3 else -1e9
        for j in range(i):
            m=cands[j][0]
            if m<n<=m+maxstep and best[j]>-1e8:
                sc=best[j]+w-(0.3*(n-m-1))
                if sc>best[i]: best[i]=sc;prev[i]=j
    i=max(range(len(cands)),key=lambda k:best[k]); out=[]
    if best[i]<0: return []
    while i!=-1: out.append(i); i=prev[i]
    return out[::-1]

def split_inline_numbered(t):
    ms=[m for m in re.finditer(r'(?<!Section)(?<!Sections)(?<!Rule)(?<!Article)(?<!Order)(?<!No)(?<!Nos)(?<!and)(?<!to)(?<!dated)(?<!para)(?<!paras)(?<!,)\s+(\d{1,3})\.\s+(?=["“(]?[A-Z])',t)]
    keep=best_chain([(int(m.group(1)),1.0) for m in ms],maxstep=2)
    paras=[];pos=0;cur_no=None
    for k in keep:
        m=ms[k]; seg=t[pos:m.start()].strip()
        if seg: paras.append((cur_no,seg))
        cur_no=int(m.group(1));pos=m.end()
    paras.append((cur_no,t[pos:].strip()))
    return paras

cases=[];report=[]
for cid in ORDER:
    x=raw[cid]; B=x['blocks']; note=[]
    first=B[0]['text']; old='PETITIONER' in first[:400]
    headnote='';act='';preamble=''
    if cid==1452565:
        p=B[0]['text']; s=re.search(r'This\s+appeal\s+arises',p); e=p.find('#',s.start()) if s else -1
        # header (cause title) sits before the body in the same stream
        body=p[s.start():]; end=re.search(r'December\s+3,\s+2009\.',body); body=body[:end.end()]
        assert '#' not in body, 'junk in body'
        body=clean(body,False)
        body=re.sub(r'\s*\.J\.\s*\(TARUN CHATTERJEE\).*$','',body).strip()
        nat=[{'no':n,'text':t} for n,t in split_inline_numbered(body)]
        if nat and nat[0]['no'] is None: nat[0]['no']=1 if (len(nat)>1 and nat[1]['no']==2) else None
        note.append('source copy was a damaged Word file; readable text recovered and split on the court\'s inline paragraph numbers')
        numbered=True
    else:
        blocks=[dict(b) for b in B]
        if old:
            h=first
            m=re.search(r'HEADNOTE:?(.*?)(?:JUDGMENT:|$)',h,re.S); headnote=clean(m.group(1),True) if m else ''
            m=re.search(r'ACT:(.*?)(?:HEADNOTE:|JUDGMENT:|$)',h,re.S); act=clean(m.group(1),True) if m else ''
            blocks=blocks[1:]
        elif blocks[0]['tag']=='PRE':
            if W(blocks[0]['text'])>260: note.append('long header block kept out of paragraphs (%d words)'%W(blocks[0]['text']))
            preamble=clean(blocks[0]['text'],False); blocks=blocks[1:]
        cl=[]
        for b in blocks:
            t=clean(b['text'],old)
            if t: cl.append({'tag':b['tag'],'text':t})
        # opening procedural lines of reported judgments -> preamble
        if old:
            k=None
            for i,b in enumerate(cl[:14]):
                if re.search(r'delivered by|^[A-Z][A-Za-z. ]{2,40},\s*(C\.\s*J|J)\.?(\s|-|$)|^O\s*R\s*D\s*E\s*R|^J\s*U\s*D\s*G\s*M\s*E\s*N\s*T',b['text']): k=i;break
            if k:
                preamble=' '.join(b['text'] for b in cl[:k]); cl=cl[k:]
        # trailing signatures / reporter initials
        while cl and is_trailer(cl[-1]['text']): cl.pop()
        cl=[b for b in cl if not re.fullmatch(r'[\W_]*',b['text'])]
        # paragraph numbers
        cand=[]
        for i,b in enumerate(cl):
            b['no']=None
            m=re.match(r'^(\d{1,3})\.\s*(?=\S)',b['text'])
            if m: cand.append((i,int(m.group(1)),1.0 if b['tag']=='P' else 0.9,m.end()))
        keep=best_chain([(c[1],c[2]) for c in cand])
        numbered=len(keep)>=8
        if numbered:
            for k in keep:
                i,n,w,e=cand[k]; cl[i]['no']=n; cl[i]['text']=cl[i]['text'][e:].strip()
        nat=[]
        if numbered:
            for b in cl:
                if b['no'] is not None or not nat: nat.append({'no':b['no'],'text':b['text']})
                else: nat[-1]['text']+=' '+b['text']
            if nat and nat[0]['no'] is None and W(nat[0]['text'])<30:      # judge name line etc.
                preamble=(preamble+' '+nat[0]['text']).strip(); nat=nat[1:]
        else:
            for b in cl:
                if b['tag']=='BLOCKQUOTE' and nat: nat[-1]['text']+=' '+b['text']
                else: nat.append({'no':None,'text':b['text']})
            # merge tiny fragments forward, split unstructured giants
            merged=[]
            carry=''
            for p in nat:
                t=(carry+' '+p['text']).strip() if carry else p['text']; carry=''
                if W(t)<25: carry=t; continue
                merged.append(t)
            if carry:
                if merged: merged[-1]+=' '+carry
                else: merged=[carry]
            nat=[]
            for t in merged:
                if W(t)>600:
                    for piece in pack(t,target=220,hard=320,overlap=False): nat.append({'no':None,'text':piece})
                    note.append('one unbroken block of %d words split into sentence groups'%W(t))
                else: nat.append({'no':None,'text':t})
    footnotes=''
    while nat:
        t=nat[-1]['text']
        ms=[m for m in re.finditer(r'(?:^|\s)\[1\]\s',t)]
        if not footnotes and ms and '[2]' in t[ms[-1].end():]:
            footnotes=t[ms[-1].start():].strip(); t=t[:ms[-1].start()]
        t=re.sub(r'[\s\-_]+$','',t)
        m=re.search(r'\s*(?:[.…]+\s*)?(?:C\.?\s*)?J(?:I)?\.\s*[\[(]\s*[A-Z][^)\]]{2,40}[)\]].*$',t)
        if m and m.start()>len(t)-400: t=t[:m.start()].strip()
        t=re.sub(r'\s(?:[A-Z]\.){2,4}(?=\s+(?:Appeals?|Petitions?)\b)','',t)
        if t and t[-1] not in '.?!)"”': t+='.'
        nat[-1]['text']=t
        if t or len(nat)==1: break
        nat.pop()
    paras=[]
    for i,p in enumerate(nat,1):
        paras.append({'para_no': p['no'] if (numbered and p['no'] is not None) else i if not numbered else (paras[-1]['para_no'] if paras else 0),'text':p['text'],'word_count':W(p['text'])})
    if numbered and paras and paras[0]['para_no']==0 and (len(paras)==1 or paras[1]['para_no']>=2): paras[0]['para_no']=1
    m=re.search(r' on (\d{1,2}) ([A-Za-z]+),? (\d{4})$',x['title']); dt=datetime.datetime.strptime(' '.join(m.groups()),'%d %B %Y').date()
    bench=[b.strip() for b in re.sub(r'^Bench:\s*','',x['bench']).split(',') if b.strip()]
    bench=[re.sub(r'\s+',' ',b) for b in bench]
    cites=[c.strip() for c in re.sub(r'^Equivalent citations:\s*','',x['cite']).split(', ') if c.strip()][:5]
    case={'case_id':str(cid),'case_name':NAMES[cid],'court':'Supreme Court of India','date':dt.isoformat(),'year':dt.year,'bench':bench,
          'author':re.sub(r'^Author:\s*','',x['author']).strip() or None,'citations':cites,'category':CATMAP[CAT[cid]],
          'source_url':x['url'],'para_numbering':'court' if numbered else 'generated','statutes_noted':act or None,'headnote':headnote or None,
          'preamble':preamble or None,'footnotes':footnotes or None,'paragraph_count':len(paras),'word_count':sum(p['word_count'] for p in paras),'paragraphs':paras}
    cases.append(case); report.append((cid,NAMES[cid][:38],dt.isoformat(),'numbered' if numbered else 'generated',len(paras),case['word_count'],W(headnote),W(preamble),'; '.join(note)))

chunks=[];n=0
for c in cases:
    for p in c['paragraphs']:
        parts=pack(p['text'])
        for k,t in enumerate(parts,1):
            n+=1
            chunks.append({'id':n,'chunk_id':'%s-p%03d-%d'%(c['case_id'],p['para_no'],k) if c['para_numbering']=='court' else '%s-g%03d-%d'%(c['case_id'],p['para_no'],k),
                           'case_id':c['case_id'],'case_name':c['case_name'],'court':c['court'],'year':c['year'],'category':c['category'],
                           'para_no':p['para_no'],'part':k,'parts_in_para':len(parts),'paragraph':t,'word_count':W(t),'source_url':c['source_url']})
json.dump(cases,open(outdir+'/atl_judgments_clean.json','w'),ensure_ascii=False,indent=1)
json.dump(chunks,open(outdir+'/atl_chunks.json','w'),ensure_ascii=False,indent=1)
for r in report: print(*r,sep=' | ')
L=sorted(c['word_count'] for c in chunks); print('chunks',len(chunks),'min',L[0],'p10',L[len(L)//10],'med',L[len(L)//2],'p90',L[len(L)*9//10],'max',L[-1],'<20:',sum(l<20 for l in L),'>350:',sum(l>350 for l in L))
