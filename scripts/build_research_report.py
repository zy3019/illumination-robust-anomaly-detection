"""Build an eight-page English report and editable Markdown from accepted evidence.

Uses the bundled document Python runtime; does not change the experiment environment.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_integrity import verify_report_inputs
FORMAL=ROOT/'reports/formal_v1'
FAIL=FORMAL/'failure_analysis'
OUT=ROOT/'reports/research_report_v2'
PDF=ROOT/'output/pdf/illumination_robustness_research_report_v2.pdf'
SHA='58efbb9a2e54a4e276d062235f141f687a0aeaa84f26e48eb34cfe34b82a6885'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def one(rows,**keys):
    selected=[r for r in rows if all(r[k]==str(v) for k,v in keys.items())]
    assert len(selected)==1,(keys,len(selected))
    return selected[0]


def mean_sd(row,key,scale=1,digits=4,signed=False):
    mean=float(row[key+'_mean'])*scale;sd=float(row[key+'_std'])*scale
    return f'{mean:+.{digits}f} +/- {sd:.{digits}f}' if signed else f'{mean:.{digits}f} +/- {sd:.{digits}f}'


def main():
    verify_report_inputs(ROOT)
    acceptance=read('records/formal_v1/acceptance.json')
    assert acceptance['status']=='accepted' and acceptance['protocol_sha256']==SHA
    assert digest(FORMAL/'report.md')==acceptance['report_sha256']
    assert digest(ROOT/'records/formal_v1/protocol.json')==SHA
    provenance=read('reports/formal_v1/failure_analysis/provenance.json')
    assert provenance['protocol_sha256']==SHA
    for name,sha in provenance['outputs_sha256'].items():assert digest(FAIL/name)==sha,name
    overview=read('reports/formal_v1/overview.json')
    failure=read('reports/formal_v1/failure_analysis/overview.json')
    assert failure['selected_cases']==18 and failure['scoring_events']==98280
    macro=csv_rows(FORMAL/'macro_severity_seed_summary.csv')
    paired=csv_rows(FORMAL/'macro_severity_paired_seed_summary.csv')
    trans=csv_rows(FAIL/'paired_transitions.csv')
    manifest=read('records/manifests/mvtec_v1.json')
    pages=[]
    def page(title):
        block=[];pages.append(dict(title=title,blocks=block));return block
    def p(block,text):block.append(dict(kind='paragraph',text=text))
    def h(block,text):block.append(dict(kind='heading',text=text))
    def table(block,rows,widths=None):block.append(dict(kind='table',rows=rows,widths=widths))
    def fig(block,path,caption,width=503):block.append(dict(kind='image',path=path,caption=caption,width=width))

    b=page('Illumination Robustness with Limited Normal Training Images')
    b.append(dict(kind='subtitle',text='A controlled study of a simplified PatchCore baseline\nTechnical report | 8 October 2026 | formal_v1 results, editorial revision 2'))
    h(b,'Abstract')
    p(b,'This project examines how normal-image budgets and synthetic illumination augmentation affect industrial anomaly detection at a fixed memory-bank capacity. A frozen ResNet18 feature extractor and a 1,000-vector memory bank are evaluated on bottle, metal_nut and tile. The locked matrix contains 90 models, three seeds and 13 conditions. Its 98,280 scoring events come from 252 independent held-out images, rather than 98,280 independent examples. All previously accepted development models and their clean-calibration thresholds are reused without selecting a winning seed.')
    p(b,'Clean macro image AUROC increases with the original-image budget. However, illumination degradation is not universally monotonic: both main methods have 55 nonmonotone AUROC-degradation sequences out of 108 dependent sequences. Across 12 perturbations, three-view augmentation changes macro image AUROC by +0.00087, +0.00235, +0.00017 and -0.00038 at budgets 5, 10, 25 and all. At the full budget, mean FPR falls by 7.90 percentage points, while pixel AUROC and recall decrease slightly. A post-hoc audit retains every error and illustrates both corrected and introduced failures. These observations support conditional benefits, not universal robustness or statistical significance.')
    h(b,'1. Research questions')
    table(b,[['Question','Evidence'],['RQ1: Does a larger normal-image budget improve clean performance?','Compare 5, 10, 25 and all originals within each category and seed.'],['RQ2: Does illumination degradation depend on that budget?','Compare each perturbed image with its clean counterpart at a fixed threshold.'],['RQ3: Does augmentation help at fixed bank capacity?','Pair A/B; compare one-view C at budgets 5 and 25.']],[1.1,1.8])
    h(b,'Scope and contribution')
    p(b,'The contribution is a traceable empirical study and a checked lightweight implementation, rather than a new anomaly-detection algorithm. The deliverables include a frozen protocol, explicit data usage, paired multi-seed results, resource records, and reproducible error analysis. Negative results are retained. The split is custom; the numbers are not official full-test MVTec AD benchmark scores.')
    p(b,'Reading guide: methods on page 2; data and protocol on page 3; RQ1-RQ2 on page 4; RQ3 on page 5; post-hoc failures on pages 6-7; limitations, reproducibility and references on page 8.')

    b=page('2. Method and implementation boundary')
    p(b,'PatchCore uses a representative memory bank of nominal patch features for anomaly detection [1]. MVTec AD supplies industrial object and texture images with normal training images and annotated test defects [2,3]. This study adopts the memory-bank idea in a smaller, explicitly specified implementation. It does not attempt to match the original paper configuration or claim its benchmark performance.')
    table(b,[['Component','Locked implementation'],['Input','RGB; direct bilinear resize to 224 x 224; ImageNet normalization'],['Backbone','Frozen ImageNet ResNet18; layer2 and layer3'],['Patch features','3 x 3 average pooling; bilinear feature alignment; 384 dimensions'],['Coreset','64-dimensional Gaussian projection; greedy selection; fixed seed'],['Bank','1,000 vectors; distinct candidate indices; float32'],['Image score','Maximum raw Euclidean 1-nearest-neighbor patch distance'],['Anomaly map','Bilinear upsampling; no Gaussian smoothing'],['Threshold','Own clean-normal calibration 95th linear empirical quantile']],[1,2.8])
    p(b,'For image x with patch descriptors f_i(x) and bank M, the score is s(x) = max_i min_{m in M} ||f_i(x)-m||_2. A prediction is anomalous only when s(x) > t. The same t is used for every illumination condition of that model. The maximum of the interpolated pixel map need not equal the maximum of the original patch scores.')
    h(b,'2.1 Original images, views and paired controls')
    p(b,'A uses original normal images only. B adds three illumination views per original; C adds one. Each added view randomly chooses brightness or contrast and samples its factor uniformly from [0.8,1.2]. C uses the same first view as B. For each category and seed, original-image budgets are nested and methods share identical original images. All methods end with 1,000 memory vectors, so the bank capacity is controlled even though candidate counts differ.')
    h(b,'2.2 Numerical implementation audit')
    p(b,'Before final testing, the bottle/seed42/n5 A/B implementation was compared with actual Anomalib 2.2.0 code at a fixed commit. Features matched exactly; shared-bank k=1 image scores differed by at most 5.365e-6 in the audited cases. Original banks and scores were reconstructed exactly. The audit did not cover a full framework training pipeline or every category.')
    p(b,'The simplified implementation intentionally differs in projection/coreset details, omits the default k=9 score reweighting, and uses a different map upsampling/smoothing path. These are disclosed algorithmic differences, not silently treated as identical framework behavior. The detailed local evidence is reports/baseline_audit.md.')

    b=page('3. Data, frozen protocol and measurement')
    data=[['Category','Train pool','Calibration','Dev','Final normal','Final defect']]
    for cat in ['bottle','metal_nut','tile']:
        c=overview['counts'][cat]
        data.append([cat,c['train_pool'],c['calibration'],sum(r['category']==cat and r['split']=='dev' for r in manifest['rows']),c['normal'],c['anomalous']])
    table(b,data,[1.25,1,1,0.65,1.15,1.15])
    p(b,'The file-level manifest holds back about 20% of normal training images for calibration and about 20% of each test-defect stratum for development. Final evaluation contains 60 normal and 192 anomalous images. Development anomaly labels were used during method development; the research workflow is therefore not described as using only normal data. Training and threshold fitting themselves use normal images only.')
    h(b,'3.1 Matrix and illumination')
    p(b,'Seeds are 17, 42 and 2026. A and B cover four budgets in three categories (72 models); C covers budgets 5 and 25 (18 models). All 90 accepted models are reused. Full budget means 167, 176 and 184 originals for bottle, metal_nut and tile, respectively. No final-set tuning, seed selection or condition deletion is permitted.')
    p(b,'Before ImageNet normalization, brightness is clip(b*x,0,1). Contrast is clip(c*(x-mu)+mu,0,1), where mu is the per-image, per-channel spatial mean. The 13 conditions are clean plus separate brightness and contrast factors 0.9, 1.1, 0.7, 1.3, 0.5 and 1.5. Factors 0.5/1.5 were designated as stress conditions before final evaluation. Masks are resized by nearest neighbor and are not geometrically changed by illumination.')
    h(b,'3.2 Metrics and dependence')
    p(b,'Core metrics are image AUROC, pooled resized-pixel AUROC, FPR and anomalous-image recall. Reported counts retain FP/normal and TP/anomalous denominators. Degradation is clean AUROC minus perturbed AUROC, or perturbed FPR minus clean FPR. For method effects, subtraction is within category, budget, seed and condition. Categories and conditions are equally weighted within each seed; the three-seed mean and sample SD (ddof=1) follow. SD is not a confidence interval. Repeated conditions, images and categories are not extra independent seeds. AU-PRO is outside this matrix.')
    h(b,'3.3 Freeze and acceptance')
    p(b,'Protocol formal_v1 was locked on 7 October 2026 at 15:36:50 (Asia/Shanghai), before the first final-image exposure at 15:37:35. Exact sources, weights, models, sample lists and thresholds were hashed. The optimized evaluator matched 819 development scores/maps exactly and 39 pixel metrics with zero difference before locking. Six preselected final pixel metrics also matched the original rank algorithm exactly. All final image metrics and confusion counts were independently recomputed.')
    p(b,'The 95th empirical calibration quantile does not guarantee a future 5% FPR, particularly with only 42-46 calibration normals and 16-26 final normals per category.')

    b=page('4. Results: sample budget and degradation')
    clean=[['Original budget','A clean image AUROC','B clean image AUROC']]
    for n in ['5','10','25','all']:
        clean.append([n,*[mean_sd(one(macro,budget=n,method=m,severity='clean'),'image_auroc') for m in ['A','B']]])
    table(b,clean,[1,2,2])
    p(b,'RQ1: the clean category macro increases with budget for A and B. This does not imply uniform improvement: bottle remains at 1.0 image AUROC, metal_nut improves in seed mean, and tile decreases from 5 to 10 originals. At full budget, B has clean macro FPR 0.0646 versus A 0.0442; B pixel AUROC is 0.9638 versus A 0.9654.')
    fig(b,'reports/research_report_v1/figures/budget_summary.png','Figure 1. Equal-category macro for clean AUROC and mean stress degradation. Thin lines are seeds; bars are sample SD, not CI. The all budget differs by category. Per-category plots remain in the formal results appendix.',width=495)
    p(b,'RQ2: each method has 108 dependent category/seed/perturbation sequences across budgets. For AUROC degradation, A has 28 tied, 11 nonincreasing, 14 nondecreasing and 55 nonmonotone sequences; B has 32, 6, 15 and 55. FPR-increase sequences are nonmonotone in 37 A and 48 B cases. The data do not support a universal rule that more originals always reduce illumination degradation.')

    b=page('5. Results: augmentation at fixed capacity')
    effect=[['Budget','Image AUROC','Pixel AUROC','FPR (pp)','Recall (pp)']]
    for n in ['5','10','25','all']:
        r=one(paired,budget=n,comparison='B-A',severity='all_perturbed')
        effect.append([n,mean_sd(r,'image_auroc',digits=5,signed=True),mean_sd(r,'pixel_auroc',digits=5,signed=True),mean_sd(r,'fpr',scale=100,digits=2,signed=True),mean_sd(r,'recall',scale=100,digits=2,signed=True)])
    table(b,effect,[.55,1.6,1.6,1.25,1.25])
    p(b,'Paired B-A over all 12 perturbations and three equally weighted categories, then mean +/- sample SD over seeds. pp denotes percentage points. Negative FPR is beneficial; positive recall and AUROC are beneficial. The modest image-AUROC differences do not establish significance from three seeds.')
    fig(b,'reports/research_report_v1/figures/paired_effects.png','Figure 2. Paired effects separated into clean, mild, moderate and stress conditions; bars are three-seed sample SD. The full-budget FPR benefit coexists with pixel-AUROC and recall costs.',width=495)
    h(b,'5.1 One-view ablation and clean costs')
    p(b,'For image AUROC over all perturbations, B-C is +0.00120 +/- 0.00338 at five originals and -0.00126 +/- 0.00246 at 25 originals. More views are not reliably better. Their mean ordering reverses relative to the development results; no method was reselected after observing this. Full-budget B reduces perturbed FPR by 7.90 pp but decreases pixel AUROC by 0.00132 and recall by 0.24 pp, while increasing clean FPR by 2.04 pp. RQ3 therefore has a conditional, multi-metric answer.')

    b=page('6. Post-hoc failure analysis: corrected and introduced errors')
    p(b,'This analysis was designed after final results were visible. It reads accepted scores and saved maps only. All 98,280 events are retained, including 2,368 false positives and 4,667 false negatives. A/B contribute 39,312 paired comparisons. C is included in the error inventory but has fewer budgets, so raw cumulative counts must not rank it against A/B.')
    error_table=[['Category / error','A errors','B errors','Both wrong','B rescued','B introduced']]
    for cat in ['bottle','metal_nut','tile']:
        for label,name in [('0','FP'),('1','FN')]:
            rows=[r for r in trans if r['category']==cat and r['label']==label]
            sums=[sum(int(r[k]) for r in rows) for k in ['A_errors','B_errors','both_wrong','B_rescued','B_introduced']]
            assert sums[1]-sums[0]==sums[4]-sums[3]
            error_table.append([cat+' / '+name,*sums])
    table(b,error_table,[1.6,1,1,1.1,1.1,1.2])
    p(b,'Dependent error events across all A/B budgets, seeds and conditions, including clean. These counts are not independent sample sizes or deployment rates. The identity B errors - A errors = introduced - rescued holds in every stratum. In metal_nut, fewer missed-defect events coexist with more false alarms.')
    fig(b,'reports/research_report_v1/figures/bottle_rescued_fp.png','Figure 3. Normal bottle good/000.png, full budget, seed17. Under brightness=0.5, A crosses its threshold (1.142417), while B remains below its own threshold (0.910181). The paired pattern occurs in all three seeds.',width=503)
    p(b,'Maps are divided by their own image thresholds and share one display scale within each panel. They are not probabilities or pixel segmentation decisions. Eighteen cases cover six roles in each category. Selection prioritizes cross-seed recurrence, then stress conditions, specified budget order, frozen condition order, path and seed. All candidates and empty-role checks are retained; no role was empty.')

    b=page('7. Post-hoc failure analysis: response is not a decision')
    fig(b,'reports/research_report_v1/figures/metal_nut_persistent_fn.png','Figure 4. metal_nut bent/007.png, n=5, seed17. A/B already miss the defect under clean input. Darkening reduces the score/threshold ratios to 0.829985 and 0.843059. The resized mask occupies about 0.72% of the image.',width=503)
    p(b,'The visible hotspot near the annotated edge does not guarantee a positive image decision. This example cannot isolate defect size, backbone sensitivity or calibration as the cause. A segmentation threshold was not defined, so visual correspondence is not a segmentation-success claim.')
    fig(b,'reports/research_report_v1/figures/tile_introduced_fn.png','Figure 5. tile gray_stroke/012.png, n=25, seed17. At brightness=1.5, A detects the defect (1.017577) and B misses it (0.960810). Both already miss it under clean input.',width=503)
    p(b,'Here introduced FN means a disadvantage of B relative to A at the same condition, not a clean-to-perturbed failure newly caused in B. The all_scores table separately records that within-model comparison. Other examples include corrected and introduced decisions close to one; six decimal places prevent a ratio such as 1.000327 from appearing exactly equal to the strict threshold.')

    b=page('8. Discussion, reproducibility and limitations')
    p(b,'The strongest conclusion is the need to separate ranking quality, threshold decisions and spatial response. Full-budget bottle at contrast=1.5 reaches image AUROC 1.0 for A and B in every seed, yet A falsely flags 16/16 normals in every seed and B flags 16/16, 16/16 and 15/16. At brightness=0.5, B instead substantially reduces false alarms. A high AUROC alone does not establish usable fixed-threshold behavior.')
    p(b,'Limitations include three categories, three seeds, small normal evaluation sets, synthetic brightness/contrast, direct square resizing, one backbone and a simplified coreset/scoring path. Development anomaly labels influenced the workflow. Post-hoc visual cases are explanatory, not causal or independently confirmed evidence. Clipping co-occurs with some errors, but zero-clipping cases also fail. New normalization, calibration or resolution hypotheses require a separately declared study and new independent evaluation data.')
    h(b,'8.1 Resources and reproducibility')
    p(b,'Accepted evaluation subprocesses total 46.94 minutes and 19.22 GiB of output. Peak CPU working set is 1.29 GiB; Torch peak allocated/reserved memory is 51.8/66.0 MiB, not whole-device usage. These costs exclude historical bank construction, interrupted attempts, extra audit work and waiting time. Job 40 was interrupted and repeated under the same protocol; 1,196 scores and 13 condition metrics matched exactly, excluding timing. Its first attempt remains archived in place.')
    p(b,'Run from the repository root. Reproduction: docs/reproduction_v2.md. Frozen execution: docs/formal_execution.md. Protocol and acceptance: records/formal_v1/. Final tables: reports/formal_v1/. Reproduce diagnosis with .venv-audit/Scripts/python.exe -B scripts/analyze_final_failures.py. Report source and rendering instructions are in reports/research_report_v1/. No GPU inference is needed to rebuild this report. Exact input/output hashes accompany both analysis and report.')
    h(b,'8.2 Contribution and tool disclosure')
    p(b,'The human project owner directed the project stages and authorized protocol locking, final testing and this analysis. Code, experimental execution, numerical checks, figures and this report were produced with AI assistance in the shared workspace. This is a project-level contribution record, not evidence that the owner independently implemented or reviewed every component. Personal authorship and understanding claims require the owner\'s own verification before external submission.')
    h(b,'References and local evidence')
    p(b,'[1] K. Roth et al. Towards Total Recall in Industrial Anomaly Detection. CVPR, 2022, pp. 14318-14328. <a href="https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html">CVF paper</a>.')
    p(b,'[2] P. Bergmann et al. MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR, 2019. <a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html">CVF paper</a>. [3] <a href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad">MVTec AD official dataset</a>. Sources checked on 7 October 2026.')
    p(b,'Local evidence: docs/formal_protocol_v1.md; reports/baseline_audit.md; reports/formal_v1/report.md; failure_analysis/report.md and interpretation.md. Locked protocol SHA-256: '+SHA+'.')

    OUT.mkdir(parents=True,exist_ok=True);PDF.parent.mkdir(parents=True,exist_ok=True)
    (OUT/'report_content.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
    md=[]
    for index,page in enumerate(pages,1):
        md += ['# '+page['title'],'',f'<!-- PDF page {index} -->','']
        for block in page['blocks']:
            if block['kind'] in ['paragraph','subtitle']:md += [block['text'],'']
            elif block['kind']=='heading':md += ['## '+block['text'],'']
            elif block['kind']=='table':
                rows=block['rows'];md+=['| '+' | '.join(map(str,rows[0]))+' |','|'+'|'.join(['---']*len(rows[0]))+'|']
                md+=['| '+' | '.join(map(str,row))+' |' for row in rows[1:]];md+=['']
            else:md += [f"![{block['caption']}]({os.path.relpath(ROOT/block['path'],OUT).replace(os.sep,'/')})",'',block['caption'],'']
    (OUT/'report.md').write_text('\n'.join(md),encoding='utf-8')
    styles={
        'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=13.6,spaceAfter=9,textColor=colors.HexColor('#263544')),
        'heading':ParagraphStyle('heading',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=5,spaceAfter=7,textColor=colors.HexColor('#164b65')),
        'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=20,leading=24,spaceAfter=13,textColor=colors.HexColor('#103649')),
        'subtitle':ParagraphStyle('subtitle',fontName='Helvetica',fontSize=11,leading=16,spaceAfter=15,textColor=colors.HexColor('#607180')),
        'caption':ParagraphStyle('caption',fontName='Helvetica',fontSize=8.3,leading=11,spaceBefore=3,spaceAfter=9,textColor=colors.HexColor('#54636c')),
        'cell':ParagraphStyle('cell',fontName='Helvetica',fontSize=8.5,leading=11,textColor=colors.HexColor('#263544')),
        'cellhead':ParagraphStyle('cellhead',fontName='Helvetica-Bold',fontSize=8.5,leading=11,textColor=colors.white)
    }
    story=[]
    for index,page in enumerate(pages):
        if index:story.append(PageBreak())
        story.append(Paragraph(escape(page['title']),styles['title']))
        for block in page['blocks']:
            kind=block['kind']
            if kind in ['paragraph','heading','subtitle']:
                txt=block['text'] if '<a href=' in block['text'] else escape(block['text'])
                story.append(Paragraph(txt.replace('\n','<br/>'),styles['body' if kind=='paragraph' else kind]))
            elif kind=='table':
                data=[[Paragraph(escape(str(cell)),styles['cellhead' if i==0 else 'cell']) for cell in row] for i,row in enumerate(block['rows'])]
                weights=block['widths'] or [1]*len(data[0]);widths=[503*w/sum(weights) for w in weights]
                t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#164b65')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf4f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c4d7df'))]))
                story.extend([t,Spacer(1,9)])
            else:
                path=ROOT/block['path']
                with PILImage.open(path) as im:w,hp=im.size
                width=block['width'];story.append(Image(str(path),width=width,height=width*hp/w))
                story.append(Paragraph(escape(block['caption']),styles['caption']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c4d7df'));canvas.line(46,39,A4[0]-46,39)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607180'))
        canvas.drawString(46,26,'LIMITED NORMAL DATA  /  ILLUMINATION ROBUSTNESS  /  formal_v1')
        canvas.drawRightString(A4[0]-46,26,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(PDF),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=39,bottomMargin=49,
                          title='Illumination Robustness with Limited Normal Training Images',author='AI-assisted project technical report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(PDF);texts=[p.extract_text() for p in reader.pages]
    assert len(reader.pages)==8, f'Expected 8 pages, got {len(reader.pages)}'
    assert all(len(t)>250 for t in texts), 'Unexpected near-empty page'
    for i,page in enumerate(pages):assert page['title'] in texts[i].replace('\n',' '),(i,page['title'])
    (OUT/'pdf_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    paths=[FORMAL/name for name in ['report.md','macro_severity_seed_summary.csv','macro_severity_paired_seed_summary.csv','seed_stability.png','paired_effects.png','overview.json']]
    paths += [FAIL/name for name in ['provenance.json','paired_transitions.csv','interpretation.md','bottle_rescued_fp.png','metal_nut_persistent_fn.png','tile_introduced_fn.png']]
    paths += [ROOT/'scripts/build_research_report.py',ROOT/'records/formal_v1/protocol.json',ROOT/'records/manifests/mvtec_v1.json']
    paths += list((OUT/'figures').glob('*.png')) + [ROOT/'scripts/render_research_case_figures.py',FORMAL/'macro_severity_by_seed.csv']
    evidence=dict(protocol_sha256=SHA,report_date='2026-10-08',pages=len(reader.pages),
                  inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                  outputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [PDF,OUT/'report.md',OUT/'report_content.json']},
                  external_references_checked='2026-10-07',visual_review='pending')
    (OUT/'provenance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(dict(pdf=str(PDF),pages=len(reader.pages),page_text_characters=[len(t) for t in texts])))


if __name__=='__main__':main()

