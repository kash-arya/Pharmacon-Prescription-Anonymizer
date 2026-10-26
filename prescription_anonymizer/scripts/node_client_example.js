/**
 * Pharmacon Prescription Anonymizer - Node.js Integration Client Example
 * 
 * Works with native Node.js 18+ (Fetch & FormData built-in) or axios/form-data.
 * Teammates can copy-paste this helper directly into their Express.js backend.
 */

const fs = require('fs');
const path = require('path');

const ANONYMIZER_URL = process.env.NEXT_PUBLIC_ANONYMIZER_API_URL || process.env.VITE_ANONYMIZER_API_URL || process.env.ANONYMIZER_URL || 'http://localhost:8000';

/**
 * Check if the Python Anonymizer Microservice is alive and healthy.
 */
async function checkServiceHealth() {
  try {
    const res = await fetch(`${ANONYMIZER_URL}/health`);
    const data = await res.json();
    console.log('Pharmacon Microservice Status:', data);
    return data.status === 'healthy' || data.status === 'ok';
  } catch (err) {
    console.error('Anonymizer service is unreachable:', err.message);
    return false;
  }
}

/**
 * Anonymize a prescription and stream back the sanitized file binary.
 * @param {string} inputFilePath - Path to incoming prescription (PDF/JPG/PNG)
 * @param {string} outputFilePath - Path where anonymized file should be written
 */
async function anonymizeFileStream(inputFilePath, outputFilePath) {
  const fileBuffer = fs.readFileSync(inputFilePath);
  const fileName = path.basename(inputFilePath);

  const formData = new FormData();
  formData.append('file', new Blob([fileBuffer]), fileName);

  const response = await fetch(`${ANONYMIZER_URL}/api/v1/anonymize?return_type=file`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Anonymization failed (${response.status}): ${errText}`);
  }

  const redactedCount = response.headers.get('x-redacted-count');
  const formatType = response.headers.get('x-format-type');
  console.log(`Successfully redacted ${redactedCount} entities (Format: ${formatType})`);

  const arrayBuffer = await response.arrayBuffer();
  fs.writeFileSync(outputFilePath, Buffer.from(arrayBuffer));
  console.log(`Saved sanitized document to: ${outputFilePath}`);
}

/**
 * Anonymize a prescription and receive complete JSON audit logs + Base64 preview for the website frontend.
 * @param {string} inputFilePath
 * @returns {Promise<Object>} { status, total_entities_redacted, redacted_targets, data_uri }
 */
async function anonymizeWithAuditJSON(inputFilePath) {
  const fileBuffer = fs.readFileSync(inputFilePath);
  const fileName = path.basename(inputFilePath);

  const formData = new FormData();
  formData.append('file', new Blob([fileBuffer]), fileName);

  const response = await fetch(`${ANONYMIZER_URL}/api/v1/anonymize?return_type=json`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Anonymization failed (${response.status}): ${errText}`);
  }

  const auditReport = await response.json();
  console.log(`Audit Summary: Masked ${auditReport.total_entities_redacted} entities.`);
  return auditReport;
}

/**
 * Anonymize a prescription and receive high-res JPEG previews for interactive split-sliders.
 * @param {string} inputFilePath 
 * @param {number} pageNum
 * @returns {Promise<Object>} { filename, format_type, engine, total_entities_redacted, redacted_targets, original_image, redacted_image }
 */
async function anonymizePreview(inputFilePath, pageNum = 0) {
  const fileBuffer = fs.readFileSync(inputFilePath);
  const fileName = path.basename(inputFilePath);

  const formData = new FormData();
  formData.append('file', new Blob([fileBuffer]), fileName);

  const response = await fetch(`${ANONYMIZER_URL}/api/v1/anonymize/preview?page=${pageNum}`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Preview generation failed (${response.status}): ${errText}`);
  }

  return await response.json();
}

/**
 * Anonymize an entire ZIP archive of prescriptions in a single call.
 * @param {string} zipInputPath - Path to input .zip file
 * @param {string} zipOutputPath - Path where sanitized .zip will be saved
 */
async function anonymizeBatchZip(zipInputPath, zipOutputPath) {
  const zipBuffer = fs.readFileSync(zipInputPath);
  const fileName = path.basename(zipInputPath);

  const formData = new FormData();
  formData.append('file', new Blob([zipBuffer]), fileName);

  const response = await fetch(`${ANONYMIZER_URL}/api/v1/anonymize/batch`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Batch anonymization failed (${response.status}): ${errText}`);
  }

  const totalDocs = response.headers.get('x-total-documents');
  const totalRedactions = response.headers.get('x-total-entities-redacted');
  console.log(`Batch complete: Processed ${totalDocs} documents, ${totalRedactions} entities redacted.`);

  const arrayBuffer = await response.arrayBuffer();
  fs.writeFileSync(zipOutputPath, Buffer.from(arrayBuffer));
  console.log(`Saved sanitized ZIP archive to: ${zipOutputPath}`);
}

// Module export for teammate
module.exports = {
  checkServiceHealth,
  anonymizeFileStream,
  anonymizeWithAuditJSON,
  anonymizePreview,
  anonymizeBatchZip
};

/**
 * ============================================================================
 * VERCEL / NEXT.JS / REACT FRONTEND INTEGRATION COMPONENT EXAMPLE
 * ============================================================================
 * 
 * Below is a production React component demonstrating:
 * 1. File Upload (PDF, JPG, PNG, WEBP)
 * 2. Left (Original) vs Right (Anonymized) side-by-side inspection
 * 3. Instant client-side download in original file format using `download_file_base64`
 * 
 * ```tsx
 * import React, { useState } from 'react';
 * 
 * interface PreviewResult {
 *   filename: string;
 *   anonymized_filename: string;
 *   mime_type: string;
 *   format_type: string;
 *   jurisdiction: string;
 *   total_entities_redacted: number;
 *   confidence_score: number;
 *   triage_status: 'AUTO_APPROVED' | 'NEEDS_HUMAN_REVIEW';
 *   review_reasons: string[];
 *   original_image_base64: string;
 *   redacted_image_base64: string;
 *   side_by_side_image_base64: string;
 *   download_file_base64: string;
 * }
 * 
 * export function PrescriptionAnonymizerViewer() {
 *   const [loading, setLoading] = useState(false);
 *   const [result, setResult] = useState<PreviewResult | null>(null);
 *   const [error, setError] = useState<string | null>(null);
 * 
 *   const apiUrl = process.env.NEXT_PUBLIC_ANONYMIZER_API_URL || 'http://localhost:8000';
 * 
 *   async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
 *     const file = e.target.files?.[0];
 *     if (!file) return;
 * 
 *     setLoading(true);
 *     setError(null);
 * 
 *     const fd = new FormData();
 *     fd.append('file', file);
 * 
 *     try {
 *       const res = await fetch(`${apiUrl}/anonymize/preview`, {
 *         method: 'POST',
 *         body: fd,
 *       });
 *       if (!res.ok) throw new Error(`Server returned ${res.status}: ${await res.text()}`);
 *       const data: PreviewResult = await res.json();
 *       setResult(data);
 *     } catch (err: any) {
 *       setError(err.message || 'Anonymization request failed');
 *     } finally {
 *       setLoading(false);
 *     }
 *   }
 * 
 *   function triggerDownload() {
 *     if (!result?.download_file_base64) return;
 *     const a = document.createElement('a');
 *     a.href = result.download_file_base64;
 *     a.download = result.anonymized_filename;
 *     document.body.appendChild(a);
 *     a.click();
 *     document.body.removeChild(a);
 *   }
 * 
 *   return (
 *     <div className="p-6 max-w-6xl mx-auto space-y-6">
 *       <div className="flex items-center justify-between border-b pb-4">
 *         <h1 className="text-2xl font-bold">Pharmacon Medical Prescription Anonymizer</h1>
 *         <input type="file" accept=".pdf,.png,.jpg,.jpeg,.webp" onChange={handleUpload} disabled={loading} />
 *       </div>
 * 
 *       {loading && <p className="text-blue-600 font-medium">De-identifying document & generating visual previews...</p>}
 *       {error && <p className="text-red-600 font-medium">{error}</p>}
 * 
 *       {result && (
 *         <div className="space-y-6">
 *           <div className="flex items-center justify-between bg-slate-100 p-4 rounded-lg">
 *             <div className="space-x-4">
 *               <span className="font-semibold">{result.filename}</span>
 *               <span className="text-sm px-2 py-1 bg-slate-200 rounded">{result.format_type}</span>
 *               <span className="text-sm px-2 py-1 bg-slate-200 rounded">Jurisdiction: {result.jurisdiction}</span>
 *               <span className="text-sm px-2 py-1 bg-blue-100 text-blue-800 rounded">
 *                 Redacted: {result.total_entities_redacted} entities
 *               </span>
 *               <span className={`text-sm px-2 py-1 rounded font-bold ${
 *                 result.triage_status === 'AUTO_APPROVED' ? 'bg-green-100 text-green-800' : 'bg-amber-100 text-amber-800'
 *               }`}>
 *                 {result.triage_status} ({Math.round(result.confidence_score * 100)}%)
 *               </span>
 *             </div>
 *             <button
 *               onClick={triggerDownload}
 *               className="px-4 py-2 bg-emerald-600 text-white rounded font-medium hover:bg-emerald-700"
 *             >
 *               Download in Original Format ({result.mime_type})
 *             </button>
 *           </div>
 * 
 *           <div className="grid grid-cols-2 gap-6">
 *             <div className="border rounded-lg p-3 bg-white shadow-sm">
 *               <h2 className="text-lg font-semibold text-slate-700 mb-2">Original Prescription (Left)</h2>
 *               <img src={result.original_image_base64} alt="Original" className="w-full object-contain border rounded" />
 *             </div>
 *             <div className="border rounded-lg p-3 bg-white shadow-sm">
 *               <h2 className="text-lg font-semibold text-emerald-700 mb-2">Anonymized Prescription (Right)</h2>
 *               <img src={result.redacted_image_base64} alt="Anonymized" className="w-full object-contain border rounded" />
 *             </div>
 *           </div>
 *         </div>
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */

// Standalone execution demo
if (require.main === module) {
  (async () => {
    console.log('====================================================');
    console.log('Pharmacon Microservice: Node.js Client Smoke Test');
    console.log('====================================================');
    console.log(`Connecting to: ${ANONYMIZER_URL}`);

    const isHealthy = await checkServiceHealth();
    if (!isHealthy) {
      console.log('\n[!] Microservice is not running.');
      console.log('    To start it, run: cd prescription_anonymizer && ./.venv/bin/python -m api.server');
      process.exit(1);
    }

    const samplePdf = path.join(__dirname, '..', 'data', 'samples', 'sample_digital_vector_prescription.pdf');
    if (!fs.existsSync(samplePdf)) {
      console.error(`Sample file not found at: ${samplePdf}`);
      process.exit(1);
    }

    console.log('\n[1/3] Testing JSON Audit endpoint (/api/v1/anonymize?return_type=json)...');
    const auditRes = await anonymizeWithAuditJSON(samplePdf);
    console.log(`  -> Format: ${auditRes.format_type}`);
    console.log(`  -> Redacted Entities: ${auditRes.total_entities_redacted}`);

    console.log('\n[2/3] Testing Visual Preview endpoint (/api/v1/anonymize/preview)...');
    const previewRes = await anonymizePreview(samplePdf, 0);
    console.log(`  -> Original image data-URI length: ${previewRes.original_image.length} chars`);
    console.log(`  -> Redacted image data-URI length: ${previewRes.redacted_image.length} chars`);

    console.log('\n[3/3] Testing Binary Stream endpoint (/api/v1/anonymize?return_type=file)...');
    const tmpOut = path.join('/tmp', 'node_test_anonymized.pdf');
    await anonymizeFileStream(samplePdf, tmpOut);
    if (fs.existsSync(tmpOut)) {
      const stats = fs.statSync(tmpOut);
      console.log(`  -> Sanitized file written successfully (${stats.size} bytes).`);
      fs.unlinkSync(tmpOut);
    }

    console.log('\n====================================================');
    console.log('✓ All Node.js integration tests passed successfully!');
    console.log('====================================================');
  })();
}
