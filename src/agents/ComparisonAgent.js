/**
 * ComparisonAgent.js
 * Comparison and report generation agent for clinical consultation.
 *
 * Generates before/after comparison data and full consultation reports
 * with structured sections for clinical documentation.
 */

import { CLINICAL_ZONES } from './NLUAgent.js';
import { PROCEDURE_DATABASE, PROCEDURE_TYPE } from './MedicalAdvisorAgent.js';

// ---------------------------------------------------------------------------
// Report section templates
// ---------------------------------------------------------------------------
const REPORT_SECTIONS = {
  HEADER:       'header',
  PATIENT_INFO: 'patient_info',
  CONSULTATION_SUMMARY: 'consultation_summary',
  REGIONS_ANALYZED: 'regions_analyzed',
  CHANGES_APPLIED: 'changes_applied',
  PROCEDURE_RECOMMENDATIONS: 'procedure_recommendations',
  RISK_ASSESSMENT: 'risk_assessment',
  TREATMENT_PLAN: 'treatment_plan',
  BEFORE_AFTER: 'before_after',
  NOTES: 'notes',
  DISCLAIMER: 'disclaimer',
};

// ---------------------------------------------------------------------------
// Utility: magnitude calculation
// ---------------------------------------------------------------------------
function computeMagnitude(change) {
  if (!change) return 0;
  const d = change.displacement || { x: 0, y: 0, z: 0 };
  return Math.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2) + Math.abs(change.inflate || 0);
}

function dominantAxis(change) {
  if (!change) return 'none';
  const d = change.displacement || { x: 0, y: 0, z: 0 };
  const inf = Math.abs(change.inflate || 0);
  const axes = [
    { name: 'lateral (x)', value: Math.abs(d.x) },
    { name: 'vertical (y)', value: Math.abs(d.y) },
    { name: 'depth (z)', value: Math.abs(d.z) },
    { name: 'volume (inflate)', value: inf },
  ];
  axes.sort((a, b) => b.value - a.value);
  return axes[0].value > 0 ? axes[0].name : 'none';
}

function directionDescription(change) {
  if (!change) return 'no change';
  const d = change.displacement || { x: 0, y: 0, z: 0 };
  const parts = [];
  if (Math.abs(d.x) > 0.001) parts.push(d.x > 0 ? 'laterally outward' : 'laterally inward');
  if (Math.abs(d.y) > 0.001) parts.push(d.y > 0 ? 'upward' : 'downward');
  if (Math.abs(d.z) > 0.001) parts.push(d.z > 0 ? 'forward' : 'backward');
  if (Math.abs(change.inflate || 0) > 0.001) {
    parts.push(change.inflate > 0 ? 'increased volume' : 'decreased volume');
  }
  return parts.length > 0 ? parts.join(', ') : 'no change';
}

// ---------------------------------------------------------------------------
// ComparisonAgent class
// ---------------------------------------------------------------------------
export class ComparisonAgent {
  constructor() {
    this._procedureDB = PROCEDURE_DATABASE;
    this._regionIndex = this._buildRegionIndex();
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Generate a structured before/after comparison.
   * @param {Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>} morphState
   *   Current morph state keyed by region.
   * @param {string[]} regions  Array of affected region names.
   * @returns {{ summary: object, regionDetails: Array<object>, totalMagnitude: number, timestamp: string }}
   */
  generateComparison(morphState, regions) {
    const timestamp = new Date().toISOString();
    const regionDetails = [];
    let totalMagnitude = 0;

    for (const region of regions) {
      const change = morphState[region];
      if (!change) continue;

      const mag = computeMagnitude(change);
      totalMagnitude += mag;

      regionDetails.push({
        region,
        before: { displacement: { x: 0, y: 0, z: 0 }, inflate: 0 },
        after: {
          displacement: { ...change.displacement },
          inflate: change.inflate,
        },
        magnitude: Math.round(mag * 1000) / 1000,
        magnitudePercent: Math.round(mag * 100),
        dominantAxis: dominantAxis(change),
        direction: directionDescription(change),
        category: this._categorizeRegion(region),
      });
    }

    // Sort by magnitude descending
    regionDetails.sort((a, b) => b.magnitude - a.magnitude);

    // Summary statistics
    const summary = {
      totalRegionsModified: regionDetails.length,
      totalRegionsAvailable: CLINICAL_ZONES.length,
      averageMagnitude: regionDetails.length > 0
        ? Math.round((totalMagnitude / regionDetails.length) * 1000) / 1000
        : 0,
      maxMagnitude: regionDetails.length > 0 ? regionDetails[0].magnitude : 0,
      maxMagnitudeRegion: regionDetails.length > 0 ? regionDetails[0].region : null,
      categoryCounts: this._countCategories(regionDetails),
    };

    return {
      summary,
      regionDetails,
      totalMagnitude: Math.round(totalMagnitude * 1000) / 1000,
      timestamp,
    };
  }

  /**
   * Generate a full clinical consultation report.
   * @param {Record<string, {displacement:{x:number,y:number,z:number}, inflate:number}>} morphState
   * @param {string[]} regions  Affected region names.
   * @param {{ name?: string, age?: number, gender?: string, concerns?: string, notes?: string }} patientInfo
   * @returns {{ sections: Array<{id:string, title:string, content:any}>, metadata: object }}
   */
  generateReport(morphState, regions, patientInfo = {}) {
    const comparison = this.generateComparison(morphState, regions);
    const procedureRecs = this._getRecommendations(morphState, regions);
    const riskAssessment = this._assessRisks(comparison, procedureRecs);
    const treatmentPlan = this._generateTreatmentPlan(comparison, procedureRecs);

    const reportId = `RPT-${Date.now().toString(36).toUpperCase()}`;
    const timestamp = new Date().toISOString();
    const dateFormatted = new Date().toLocaleDateString('en-US', {
      year: 'numeric', month: 'long', day: 'numeric',
    });

    const sections = [
      // --- HEADER ---
      {
        id: REPORT_SECTIONS.HEADER,
        title: 'Consultation Report',
        content: {
          reportId,
          date: dateFormatted,
          timestamp,
          platformVersion: '2.0.0',
          generatedBy: 'AI Facial Design Platform',
        },
      },

      // --- PATIENT INFO ---
      {
        id: REPORT_SECTIONS.PATIENT_INFO,
        title: 'Patient Information',
        content: {
          name: patientInfo.name || 'Not provided',
          age: patientInfo.age || 'Not provided',
          gender: patientInfo.gender || 'Not provided',
          primaryConcerns: patientInfo.concerns || 'Not specified',
          additionalNotes: patientInfo.notes || '',
        },
      },

      // --- CONSULTATION SUMMARY ---
      {
        id: REPORT_SECTIONS.CONSULTATION_SUMMARY,
        title: 'Consultation Summary',
        content: {
          totalModifications: comparison.summary.totalRegionsModified,
          overallIntensity: this._classifyOverallIntensity(comparison.summary.averageMagnitude),
          primaryAreaOfFocus: comparison.summary.maxMagnitudeRegion
            ? this._friendlyRegionName(comparison.summary.maxMagnitudeRegion)
            : 'None',
          narrative: this._generateNarrative(comparison, patientInfo),
        },
      },

      // --- REGIONS ANALYZED ---
      {
        id: REPORT_SECTIONS.REGIONS_ANALYZED,
        title: 'Facial Regions Analyzed',
        content: {
          modifiedRegions: comparison.regionDetails.map(rd => ({
            region: rd.region,
            friendlyName: this._friendlyRegionName(rd.region),
            category: rd.category,
            changePercent: rd.magnitudePercent,
            direction: rd.direction,
          })),
          unmodifiedRegionCount: CLINICAL_ZONES.length - comparison.summary.totalRegionsModified,
        },
      },

      // --- CHANGES APPLIED ---
      {
        id: REPORT_SECTIONS.CHANGES_APPLIED,
        title: 'Detailed Changes Applied',
        content: {
          changes: comparison.regionDetails.map(rd => ({
            region: rd.region,
            friendlyName: this._friendlyRegionName(rd.region),
            beforeState: 'Baseline (no modification)',
            afterState: `${rd.direction} at ${rd.magnitudePercent}% intensity`,
            displacement: rd.after.displacement,
            inflate: rd.after.inflate,
            dominantAxis: rd.dominantAxis,
          })),
          summary: `${comparison.summary.totalRegionsModified} regions modified with ` +
                   `average intensity of ${Math.round(comparison.summary.averageMagnitude * 100)}%.`,
        },
      },

      // --- PROCEDURE RECOMMENDATIONS ---
      {
        id: REPORT_SECTIONS.PROCEDURE_RECOMMENDATIONS,
        title: 'Procedure Recommendations',
        content: {
          injectable: procedureRecs.filter(r => r.type === PROCEDURE_TYPE.INJECTABLE).map(r => ({
            name: r.name,
            feasibility: r.feasibility,
            costRange: r.costRange,
            recovery: r.recoveryTime,
            permanence: r.permanence,
            rationale: r.rationale,
          })),
          surgical: procedureRecs.filter(r => r.type === PROCEDURE_TYPE.SURGICAL).map(r => ({
            name: r.name,
            feasibility: r.feasibility,
            costRange: r.costRange,
            recovery: r.recoveryTime,
            permanence: r.permanence,
            rationale: r.rationale,
          })),
          treatments: procedureRecs.filter(r => r.type === PROCEDURE_TYPE.TREATMENT).map(r => ({
            name: r.name,
            feasibility: r.feasibility,
            costRange: r.costRange,
            recovery: r.recoveryTime,
            permanence: r.permanence,
            rationale: r.rationale,
          })),
          totalOptions: procedureRecs.length,
        },
      },

      // --- RISK ASSESSMENT ---
      {
        id: REPORT_SECTIONS.RISK_ASSESSMENT,
        title: 'Risk Assessment',
        content: riskAssessment,
      },

      // --- TREATMENT PLAN ---
      {
        id: REPORT_SECTIONS.TREATMENT_PLAN,
        title: 'Suggested Treatment Plan',
        content: treatmentPlan,
      },

      // --- BEFORE / AFTER ---
      {
        id: REPORT_SECTIONS.BEFORE_AFTER,
        title: 'Before / After Comparison Data',
        content: {
          comparisonData: comparison,
          note: 'Visual comparison should be generated from the 3D morph state. ' +
                'The data above provides the numeric displacement values for each region.',
        },
      },

      // --- NOTES ---
      {
        id: REPORT_SECTIONS.NOTES,
        title: 'Additional Notes',
        content: {
          clinicianNotes: '',
          patientPreferences: patientInfo.concerns || '',
          followUpRecommended: comparison.summary.averageMagnitude > 0.2,
        },
      },

      // --- DISCLAIMER ---
      {
        id: REPORT_SECTIONS.DISCLAIMER,
        title: 'Disclaimer',
        content: {
          text: 'DISCLAIMER: This report is generated by an AI visualization tool for educational ' +
                'and consultation planning purposes only. It does NOT constitute medical advice, ' +
                'diagnosis, or treatment recommendations. Results shown are approximate visualizations ' +
                'and may not reflect actual surgical or procedural outcomes. Always consult with a ' +
                'board-certified plastic surgeon, dermatologist, or qualified medical professional ' +
                'before undergoing any procedure. Individual results vary based on anatomy, skin ' +
                'quality, healing response, and many other factors.',
          generatedAt: timestamp,
        },
      },
    ];

    return {
      sections,
      metadata: {
        reportId,
        timestamp,
        version: '2.0.0',
        totalSections: sections.length,
        regionsModified: comparison.summary.totalRegionsModified,
        proceduresRecommended: procedureRecs.length,
      },
    };
  }

  /**
   * Export report to a simplified text format.
   * @param {object} report  Report object from generateReport.
   * @returns {string}
   */
  exportToText(report) {
    const lines = [];
    const divider = '='.repeat(60);

    for (const section of report.sections) {
      lines.push(divider);
      lines.push(`  ${section.title.toUpperCase()}`);
      lines.push(divider);
      lines.push('');

      if (typeof section.content === 'string') {
        lines.push(section.content);
      } else if (section.content.text) {
        lines.push(section.content.text);
      } else if (section.content.narrative) {
        lines.push(section.content.narrative);
        if (section.content.overallIntensity) {
          lines.push(`Overall Intensity: ${section.content.overallIntensity}`);
        }
      } else if (section.content.changes) {
        for (const change of section.content.changes) {
          lines.push(`  - ${change.friendlyName}: ${change.afterState}`);
        }
        if (section.content.summary) {
          lines.push('');
          lines.push(section.content.summary);
        }
      } else if (section.content.injectable || section.content.surgical) {
        const allRecs = [
          ...(section.content.injectable || []),
          ...(section.content.surgical || []),
          ...(section.content.treatments || []),
        ];
        for (const rec of allRecs) {
          lines.push(`  - ${rec.name}`);
          lines.push(`    Feasibility: ${(rec.feasibility * 100).toFixed(0)}%`);
          lines.push(`    Cost: $${rec.costRange.min}-$${rec.costRange.max}`);
          lines.push(`    Recovery: ${rec.recovery}`);
          lines.push('');
        }
      } else if (section.content.phases) {
        for (const phase of section.content.phases) {
          lines.push(`  Phase ${phase.phase}: ${phase.title}`);
          lines.push(`    Timeline: ${phase.timeline}`);
          for (const proc of phase.procedures) {
            lines.push(`      - ${proc}`);
          }
          lines.push('');
        }
      } else if (section.content.overallRisk !== undefined) {
        lines.push(`  Overall Risk Level: ${section.content.overallRisk}`);
        if (section.content.factors) {
          for (const f of section.content.factors) {
            lines.push(`  - ${f}`);
          }
        }
      } else {
        // Generic key-value output
        for (const [key, val] of Object.entries(section.content)) {
          if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
            lines.push(`  ${key}: ${val}`);
          }
        }
      }

      lines.push('');
    }

    return lines.join('\n');
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  /** @private */
  _buildRegionIndex() {
    const index = new Map();
    for (const proc of this._procedureDB) {
      for (const region of proc.relatedRegions) {
        if (!index.has(region)) index.set(region, []);
        index.get(region).push(proc);
      }
    }
    return index;
  }

  /** @private */
  _categorizeRegion(region) {
    if (region.includes('lip') || region.includes('perioral') || region.includes('cupids') || region.includes('philtrum')) return 'Lips & Perioral';
    if (region.includes('nasal') || region.includes('nostril')) return 'Nose';
    if (region.includes('cheek') || region.includes('nasolabial')) return 'Cheeks & Mid-face';
    if (region.includes('jaw') || region.includes('chin') || region.includes('under_chin') || region.includes('marionette')) return 'Lower Face & Jaw';
    if (region.includes('brow') || region.includes('forehead') || region.includes('glabella') || region.includes('temple')) return 'Upper Face & Forehead';
    if (region.includes('eyelid') || region.includes('crow') || region.includes('eye')) return 'Eyes';
    if (region.includes('neck')) return 'Neck';
    if (region.includes('ear')) return 'Ears';
    return 'Other';
  }

  /** @private */
  _countCategories(regionDetails) {
    const counts = {};
    for (const rd of regionDetails) {
      counts[rd.category] = (counts[rd.category] || 0) + 1;
    }
    return counts;
  }

  /** @private */
  _friendlyRegionName(region) {
    const map = {
      forehead_center: 'Central Forehead',
      forehead_left: 'Left Forehead',
      forehead_right: 'Right Forehead',
      glabella: 'Glabella (Between Brows)',
      brow_left: 'Left Brow',
      brow_right: 'Right Brow',
      temple_left: 'Left Temple',
      temple_right: 'Right Temple',
      upper_eyelid_left: 'Left Upper Eyelid',
      upper_eyelid_right: 'Right Upper Eyelid',
      lower_eyelid_left: 'Left Lower Eyelid',
      lower_eyelid_right: 'Right Lower Eyelid',
      crow_feet_left: "Left Crow's Feet",
      crow_feet_right: "Right Crow's Feet",
      nasal_bridge: 'Nasal Bridge',
      nasal_dorsum: 'Nasal Dorsum',
      nasal_tip: 'Nasal Tip',
      nostril_left: 'Left Nostril',
      nostril_right: 'Right Nostril',
      nasal_ala_left: 'Left Nasal Ala',
      nasal_ala_right: 'Right Nasal Ala',
      cheek_left: 'Left Cheek',
      cheek_right: 'Right Cheek',
      cheekbone_left: 'Left Cheekbone',
      cheekbone_right: 'Right Cheekbone',
      nasolabial_left: 'Left Nasolabial Fold',
      nasolabial_right: 'Right Nasolabial Fold',
      upper_lip_center: 'Upper Lip (Center)',
      upper_lip_left: 'Upper Lip (Left)',
      upper_lip_right: 'Upper Lip (Right)',
      lower_lip_center: 'Lower Lip (Center)',
      lower_lip_left: 'Lower Lip (Left)',
      lower_lip_right: 'Lower Lip (Right)',
      lip_corner_left: 'Left Lip Corner',
      lip_corner_right: 'Right Lip Corner',
      cupids_bow: "Cupid's Bow",
      philtrum: 'Philtrum',
      marionette_left: 'Left Marionette Line',
      marionette_right: 'Right Marionette Line',
      chin_center: 'Chin (Center)',
      chin_left: 'Chin (Left)',
      chin_right: 'Chin (Right)',
      jawline_left: 'Left Jawline',
      jawline_right: 'Right Jawline',
      jaw_angle_left: 'Left Jaw Angle',
      jaw_angle_right: 'Right Jaw Angle',
      under_chin: 'Submental (Under Chin)',
      neck_left: 'Left Neck',
      neck_right: 'Right Neck',
      ear_left: 'Left Ear',
      ear_right: 'Right Ear',
      perioral: 'Perioral Area',
    };
    return map[region] || region.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  }

  /** @private */
  _classifyOverallIntensity(avgMagnitude) {
    if (avgMagnitude < 0.05) return 'Minimal';
    if (avgMagnitude < 0.10) return 'Subtle';
    if (avgMagnitude < 0.20) return 'Moderate';
    if (avgMagnitude < 0.35) return 'Noticeable';
    if (avgMagnitude < 0.50) return 'Significant';
    if (avgMagnitude < 0.70) return 'Dramatic';
    return 'Extreme';
  }

  /** @private */
  _generateNarrative(comparison, patientInfo) {
    const { summary, regionDetails } = comparison;

    if (summary.totalRegionsModified === 0) {
      return 'No modifications were applied during this consultation session.';
    }

    const categories = Object.entries(summary.categoryCounts)
      .sort((a, b) => b[1] - a[1])
      .map(([cat]) => cat);

    const intensityWord = this._classifyOverallIntensity(summary.averageMagnitude).toLowerCase();
    const topRegion = regionDetails[0];

    let narrative = `This consultation explored ${intensityWord} modifications across ` +
      `${summary.totalRegionsModified} facial region${summary.totalRegionsModified > 1 ? 's' : ''}, ` +
      `primarily in the ${categories.slice(0, 2).join(' and ')} area${categories.length > 1 ? 's' : ''}. `;

    narrative += `The most significant change was applied to the ${this._friendlyRegionName(topRegion.region)} ` +
      `(${topRegion.direction} at ${topRegion.magnitudePercent}% intensity). `;

    if (patientInfo.concerns) {
      narrative += `The patient's primary concern was: "${patientInfo.concerns}". `;
    }

    if (summary.averageMagnitude > 0.3) {
      narrative += 'The overall level of modification is significant and would likely require one or more procedures to achieve.';
    } else if (summary.averageMagnitude > 0.15) {
      narrative += 'The modifications are moderate and may be achievable with minimally invasive procedures.';
    } else {
      narrative += 'The modifications are subtle and may be achievable with conservative non-surgical approaches.';
    }

    return narrative;
  }

  /** @private */
  _getRecommendations(morphState, regions) {
    const recommendations = [];
    const seen = new Set();

    for (const region of regions) {
      const procs = this._regionIndex.get(region) || [];
      for (const proc of procs) {
        if (seen.has(proc.id)) continue;
        seen.add(proc.id);

        const overlapping = proc.relatedRegions.filter(r => regions.includes(r));
        const coverage = overlapping.length / proc.relatedRegions.length;

        let magnitudeMatch = 1.0;
        for (const r of overlapping) {
          const change = morphState[r];
          if (change) {
            const mag = computeMagnitude(change);
            magnitudeMatch = Math.min(magnitudeMatch, Math.min(proc.maxAchievableChange / Math.max(mag, 0.01), 1.0));
          }
        }

        const feasibility = Math.round((coverage * 0.5 + magnitudeMatch * 0.5) * 100) / 100;

        recommendations.push({
          id: proc.id,
          name: proc.name,
          type: proc.type,
          feasibility,
          costRange: proc.costRange,
          recoveryTime: proc.recoveryTime,
          permanence: proc.permanence,
          invasiveness: proc.invasiveness,
          rationale: `Covers ${overlapping.length} of the modified regions. ` +
                     `Feasibility: ${(feasibility * 100).toFixed(0)}%.`,
        });
      }
    }

    recommendations.sort((a, b) => b.feasibility - a.feasibility);
    return recommendations;
  }

  /** @private */
  _assessRisks(comparison, procedureRecs) {
    const factors = [];
    let riskScore = 0;

    // High-magnitude changes
    if (comparison.summary.averageMagnitude > 0.4) {
      factors.push('High overall modification intensity may require multiple sessions or combined approaches.');
      riskScore += 2;
    } else if (comparison.summary.averageMagnitude > 0.2) {
      factors.push('Moderate modification intensity; single-session procedures may be sufficient for most changes.');
      riskScore += 1;
    }

    // Many regions
    if (comparison.summary.totalRegionsModified > 10) {
      factors.push('Large number of regions modified; staged approach recommended to manage recovery.');
      riskScore += 2;
    }

    // Surgical procedures recommended
    const surgicalCount = procedureRecs.filter(r => r.type === PROCEDURE_TYPE.SURGICAL).length;
    if (surgicalCount > 2) {
      factors.push('Multiple surgical procedures may be indicated; combined surgery carries additional anesthesia and recovery risks.');
      riskScore += 3;
    } else if (surgicalCount > 0) {
      factors.push('Surgical procedure(s) recommended; standard surgical risks apply (infection, scarring, anesthesia).');
      riskScore += 2;
    }

    // Sensitive regions
    const sensitiveRegions = ['lower_eyelid_left', 'lower_eyelid_right', 'nasal_tip', 'nasal_bridge'];
    const hasSensitive = comparison.regionDetails.some(rd => sensitiveRegions.includes(rd.region));
    if (hasSensitive) {
      factors.push('Modifications include sensitive vascular regions (tear trough, nose); experienced injector recommended.');
      riskScore += 1;
    }

    if (factors.length === 0) {
      factors.push('Low overall risk. Modifications are conservative and within typical non-surgical parameters.');
    }

    let overallRisk;
    if (riskScore <= 1) overallRisk = 'Low';
    else if (riskScore <= 3) overallRisk = 'Moderate';
    else if (riskScore <= 5) overallRisk = 'Elevated';
    else overallRisk = 'High';

    return {
      overallRisk,
      riskScore,
      factors,
      recommendations: [
        'Consult with a board-certified provider before proceeding.',
        'Discuss all potential risks and complications specific to your anatomy.',
        'Consider starting with the least invasive options and reassessing.',
      ],
    };
  }

  /** @private */
  _generateTreatmentPlan(comparison, procedureRecs) {
    if (comparison.summary.totalRegionsModified === 0) {
      return {
        phases: [],
        estimatedTotalCost: { min: 0, max: 0, currency: 'USD' },
        estimatedTotalRecovery: 'N/A',
        note: 'No modifications to plan for.',
      };
    }

    // Group procedures into phases by invasiveness
    const injectables = procedureRecs.filter(r => r.type === PROCEDURE_TYPE.INJECTABLE && r.feasibility > 0.3);
    const treatments = procedureRecs.filter(r => r.type === PROCEDURE_TYPE.TREATMENT && r.feasibility > 0.3);
    const surgicals = procedureRecs.filter(r => r.type === PROCEDURE_TYPE.SURGICAL && r.feasibility > 0.3);

    const phases = [];
    let phaseNum = 1;

    // Phase 1: Non-invasive treatments (skin quality)
    if (treatments.length > 0) {
      phases.push({
        phase: phaseNum++,
        title: 'Skin Preparation & Non-Invasive Treatments',
        timeline: '1-3 months',
        procedures: treatments.slice(0, 3).map(t => `${t.name} ($${t.costRange.min}-$${t.costRange.max})`),
        notes: 'Improve skin quality before more invasive procedures.',
      });
    }

    // Phase 2: Injectables
    if (injectables.length > 0) {
      phases.push({
        phase: phaseNum++,
        title: 'Injectable Procedures',
        timeline: '1-2 sessions over 2-4 weeks',
        procedures: injectables.slice(0, 5).map(t => `${t.name} ($${t.costRange.min}-$${t.costRange.max})`),
        notes: 'Minimally invasive with quick recovery. Can be combined in single sessions where appropriate.',
      });
    }

    // Phase 3: Surgical (if needed)
    if (surgicals.length > 0) {
      phases.push({
        phase: phaseNum++,
        title: 'Surgical Procedures (If Non-Surgical Results Insufficient)',
        timeline: 'After evaluating non-surgical results; 2-6 month recovery',
        procedures: surgicals.slice(0, 3).map(t => `${t.name} ($${t.costRange.min}-$${t.costRange.max})`),
        notes: 'Surgical options should be considered only after conservative approaches. Full consultation with surgeon required.',
      });
    }

    // Phase 4: Maintenance
    phases.push({
      phase: phaseNum,
      title: 'Maintenance & Follow-Up',
      timeline: 'Ongoing (every 6-12 months)',
      procedures: ['Periodic re-evaluation', 'Touch-up injections as needed', 'Skin maintenance treatments'],
      notes: 'Injectable results are temporary and will require maintenance sessions.',
    });

    // Calculate estimated cost range
    const allRecs = [...injectables.slice(0, 5), ...treatments.slice(0, 3), ...surgicals.slice(0, 3)];
    const totalMin = allRecs.reduce((s, r) => s + r.costRange.min, 0);
    const totalMax = allRecs.reduce((s, r) => s + r.costRange.max, 0);

    return {
      phases,
      estimatedTotalCost: { min: totalMin, max: totalMax, currency: 'USD' },
      estimatedTotalRecovery: surgicals.length > 0
        ? '4-12 weeks (surgical recovery dominates)'
        : injectables.length > 0
          ? '1-2 weeks (injectable recovery)'
          : '3-7 days (treatment recovery)',
      note: 'This is a suggested sequence. Your provider will customize the plan based on your specific anatomy and goals.',
    };
  }
}

export default ComparisonAgent;
