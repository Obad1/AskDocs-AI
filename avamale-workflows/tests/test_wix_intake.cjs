// Offline tests only. No account connections or outbound emails.
const assert = require("node:assert/strict");
const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");

const source = fs.readFileSync(path.join(__dirname, "../integrations/wix_email_to_sheets.gs"), "utf8");
const env = {
  Logger: {log() {}},
  GmailApp: {search() {return []}},
  SpreadsheetApp: {},
  LockService: {}
};
vm.createContext(env);
vm.runInContext(source, env, {filename: "wix_email_to_sheets.gs"});

const email = [
"A site visitor just submitted your form AVAMALE | Sample and Service Enquiries on AVAMALE Studio",
"Submission summary:",
"Your name:",
"Jane Buyer",
"Work email:",
"JANE@EXAMPLE.COM",
"Company:",
"Example Freight Co",
"What do you handle?:",
"Freight document discrepancies",
"One issue you want solved (optional):",
"Invoice quantities differ",
"[View Submissions](https://manage.wix.app/example)"
].join("\n\n");

const result = env.avamParse_(email);
assert.equal(result.fullName, "Jane Buyer");
assert.equal(result.email, "jane@example.com");
assert.equal(result.company, "Example Freight Co");
assert.equal(result.service, "Freight document discrepancies");
assert.equal(result.problem, "Invoice quantities differ");
assert.equal(env.avamParse_("not a Wix notification"), null);
assert.equal(env.avamParse_(email.replace("Jane Buyer", "AVAMALE INTERNAL QA")), null);
assert.equal(env.avamParse_(email.replace("JANE@EXAMPLE.COM", "invalid-email")), null);

const messages = [{
  getSubject: () => "AVAMALE | Sample and Service Enquiries got a new submission",
  getFrom: () => "Wix Forms <forms@wix.com>",
  getPlainBody: () => email,
  getId: () => "wix-msg-1",
  getDate: () => new Date("2026-11-08T00:00:00Z"),
}];
const bad = [{...messages[0],getFrom: () => "Imposter <phishing@example.com>"}];
env.GmailApp.search = () => [{getMessages:()=>messages.concat(bad)}];
assert.equal(env.avamSearchMessages_().length, 1,"must reject untrusted sender");
assert.equal(env.wixImportPreview()[0].parseable,true);
console.log("Wix email parsing, QA exclusion, sender filtering, preview checks passed");
