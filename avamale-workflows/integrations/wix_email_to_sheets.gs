/**
 * AVAMALE Wix -> Google Sheets notification intake (Apps Script, bound to Lead Engine).
 * NO OUTBOUND MESSAGES. Never schedules prospecting. Imports Wix email notifications
 * as UNVERIFIED until a human checks matching Wix dashboard submission.
 *
 * Setup: (1) Set 'PAUSE' TRUE, 'ARMED' FALSE, 'DRY_RUN' TRUE on Config.
 * (2) In Extensions > Apps Script, install this code, then run wixImportPreview.
 * (3) Verify Wix notification sender and field labels in Gmail.
 * (4) Run wixInstallIntakeTrigger once, authorize, then wixImportNotifications.
 * Owner must authorize Apps Script; uploading code cannot install a trigger.
 */
const AVAM_WIX_SUBJECT = "AVAMALE | Sample and Service Enquiries got a new submission";
const AVAM_WIX_FORM = "34030b14-4829-4a50-ac05-fb067740e86c";
const AVAM_ALLOWED_WIX_SENDERS = ["wix.com", "wixautomations.com", "wixemails.com"];
const AVAM_PROHIBITED = ["INTERNAL QA", "SYNTHETIC NOTICE CHECK", "CONTROLLED TEST", "WIX-GSHEET-SYNC-QA"];
function avamHeaderMap_(sheet) {
  const a = sheet.getRange(1,1,1,sheet.getLastColumn()).getDisplayValues()[0];
  const m = {}; a.forEach((h,i) => { if(h) m[String(h).trim()] = i; });
  ["email","company","contact_name","status","notes","gmail_message_id","wix_submission_id",
   "wix_submission_status","source","created","next_action","sequence"].forEach(k => {
    if(m[k] === undefined) throw Error("Ledger missing " + k);
  });
  return m;
}
function avamExtract_(text, before, after) {
  const i = text.indexOf(before);
  if(i === -1) return "";
  const start = i + before.length;
  const j = after ? text.indexOf(after,start) : -1;
  return text.slice(start,j<0 ? undefined : j).trim();
}
function avamParse_(body) {
  const summary = String(body).split("Submission summary:").slice(1).join("Submission summary:");
  if (!summary) return null;
  const fullName = avamExtract_(summary, "Your name:", "Work email:");
  const email = avamExtract_(summary, "Work email:", "Company:");
  const company = avamExtract_(summary, "Company:", "What do you handle?:");
  const service = avamExtract_(summary, "What do you handle?:", "One issue you want solved (optional):");
  const problem = avamExtract_(summary, "One issue you want solved (optional):", "[View Submissions]");
  if (!fullName || !company || !service || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return null;
  if (AVAM_PROHIBITED.some(x => (fullName+" "+company+" "+problem).toUpperCase().includes(x))) return null;
  return {fullName:fullName.slice(0,100), email:email.trim().toLowerCase(),
          company:company.slice(0,150), service:service.slice(0,150), problem:problem.slice(0,900)};
}
function avamSearchMessages_() {
  const safeSubject = 'subject:"AVAMALE | Sample and Service Enquiries got a new submission"';
  const msgs = [];
  const threads = [];
  // Gmail search pages are capped. Never silently omit older notifications.
  for (let offset=0; offset<500; offset+=100) {
    const page=GmailApp.search(safeSubject + " newer_than:30d",offset,100);
    threads.push(...page);
    if (page.length<100) break;
    if (offset===400) throw Error("Wix notification scan exceeded 500 threads; stop and page manually");
  }
  threads.forEach(t=>t.getMessages().forEach(m=>{
    if(m.getSubject().trim() !== AVAM_WIX_SUBJECT) return;
    const from = String(m.getFrom()||"").toLowerCase();
    const match = from.match(/<([^>]+)>/) || [null,from];
    const domain = String(match[1]||"").split("@").pop();
    if(!AVAM_ALLOWED_WIX_SENDERS.includes(domain)) return;
    msgs.push(m);
  }));
  return msgs;
}
function avamSafeSheetCell_(value) {
  const s=String(value == null ? "" : value);
  // Prevent Google Sheets from interpreting untrusted form strings as formulas.
  return /^[\s\uFEFF]*[=+@-]/.test(s) ? "'" + s : s;
}
function wixImportPreview() {
  const a = avamSearchMessages_();
  const result = a.map(m=>({
    id:m.getId(), date:m.getDate().toISOString(),
    parseable:!!avamParse_(m.getPlainBody())
  }));
  Logger.log(JSON.stringify(result));
  return result;
}
function wixImportNotifications() {
  // Installing this trigger does not authorize prospecting: NO GmailApp.sendEmail.
  const lock = LockService.getScriptLock();
  if(!lock.tryLock(15000)) return "LOCKED";
  try {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sh = ss.getSheetByName("Ledger");
    if(!sh) throw Error("Missing Ledger");
    const map = avamHeaderMap_(sh);
    const end = sh.getLastRow();
    const seen = new Set();
    if(end>1) sh.getRange(2,map.gmail_message_id+1,end-1,1).getDisplayValues()
      .forEach(r=>{if(r[0])seen.add(String(r[0]));});
    const records = avamSearchMessages_();
    let added=0,duplicates=0,invalid=0;
    const sorted = records.sort((a,b)=>a.getDate()-b.getDate());
    for(const msg of sorted) {
      const msgId=msg.getId();
      if(seen.has(msgId)) {duplicates++;continue;}
      const parsed=avamParse_(msg.getPlainBody());
      if(!parsed) {invalid++;continue;}
      // User opt-outs and manually stopped contacts are never rearmed.
      const keys = sh.getLastRow()>1 ?
        sh.getRange(2,map.email+1,sh.getLastRow()-1,1).getDisplayValues() : [];
      const same = keys.some(r=>String(r[0]).trim().toLowerCase()===parsed.email);
      const row = new Array(sh.getLastColumn()).fill("");
      const put=(name,value)=>{if(map[name]!==undefined)row[map[name]]=value;};
      put("company",parsed.company); put("contact_name",parsed.fullName);
      put("email",parsed.email); put("sequence","WIX_NOTIFICATION");
      put("status",same?"MANUAL_MERGE":"INBOUND_UNVERIFIED");
      put("source","Wix owner notification email; awaiting native form ID verification");
      put("notes","Service: "+parsed.service+" | Issue: "+parsed.problem);
      put("created",msg.getDate());
      put("next_action","Verify Wix CONFIRMED form ID and consent; human reply only");
      put("gmail_message_id",msgId);
      put("wix_submission_status","NOT_VERIFIED_FROM_EMAIL");
      // The Gmail notification does not contain the Wix submission ID.
      put("wix_submission_id","");
      sh.appendRow(row.map(avamSafeSheetCell_));
      seen.add(msgId);added++;
    }
    Logger.log(JSON.stringify({added,duplicates,invalid,inspected:records.length}));
    return {added,duplicates,invalid,inspected:records.length};
  } finally {lock.releaseLock();}
}
function wixInstallIntakeTrigger() {
  const handler="wixImportNotifications";
  const existing=ScriptApp.getProjectTriggers().some(t=>t.getHandlerFunction()===handler);
  if(!existing) ScriptApp.newTrigger(handler).timeBased().everyHours(1).create();
  // Do not edit unrelated triggers. No outbound trigger is created.
  return existing?"ALREADY_INSTALLED":"INSTALLED_INTAKE_ONLY";
}
